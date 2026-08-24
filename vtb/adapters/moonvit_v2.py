from collections.abc import Iterator
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from transformers import AutoModel

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop

PATCH_SIZE = 14

# The Projector ships inside Kimi K3, not with the standalone Tower. Its three tensors
# sit in one 0.09 GB shard, so the `projected` Stage costs a small download, not surgery.
PROJECTOR_REPO = "moonshotai/Kimi-K3"
PROJECTOR_SHARD = "model-00095-of-000096.safetensors"
PROJECTOR_PREFIX = "mm_projector."


@dataclass(frozen=True)
class Preprocess:
    """PIL image to the pre-patchified tensor MoonViT-V2 expects.

    The Tower reads a packed sequence of patches, not a (3, H, W) image, so
    patchifying belongs here. At a fixed square resolution the model's own
    NaViT resize is a no-op, which `test_matches_processor` checks.
    """

    resolution: int

    def __call__(self, image) -> torch.Tensor:
        pipeline = transforms.Compose([
            square_crop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        pixels = pipeline(image)
        channels, height, width = pixels.shape
        rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
        patches = pixels.view(channels, rows, PATCH_SIZE, cols, PATCH_SIZE)
        return patches.permute(1, 3, 0, 2, 4).reshape(-1, channels, PATCH_SIZE, PATCH_SIZE)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    side = int(samples[0].shape[0] ** 0.5)
    return {
        "pixel_values": torch.cat(samples),
        "grid_thws": torch.tensor([[1, side, side]] * len(samples)),
    }


def load_projector(dtype: torch.dtype) -> nn.Module:
    """Rebuild Kimi K3's `patchmergerv2` Projector from its published weights.

    Shapes come from the checkpoint rather than a config, so a changed upstream
    checkpoint fails loudly at load_state_dict instead of quietly mismatching.
    """
    weights = load_file(hf_hub_download(PROJECTOR_REPO, PROJECTOR_SHARD))
    weights = {k.removeprefix(PROJECTOR_PREFIX): v for k, v in weights.items() if k.startswith(PROJECTOR_PREFIX)}
    width, merged_width = weights["proj.2.weight"].shape

    projector = nn.Module()
    projector.proj = nn.Sequential(
        nn.Linear(merged_width, merged_width, bias=False),
        nn.GELU(),
        nn.Linear(merged_width, width, bias=False),
    )
    projector.post_norm = nn.RMSNorm(width, eps=1e-5)
    projector.load_state_dict(weights)
    return projector.to(dtype).eval()


class MoonViTV2Adapter:
    """Kimi K3 Tower. Native-resolution ViT, run here at one fixed square size.

    The Tower emits one flat sequence for the whole batch. Every image has the same
    grid at a fixed resolution, so the sequence splits back into rows cleanly.

    Extract at batch size 1 unless flash attention is installed. Without it the model
    masks a dense square over the whole packed batch, so raising the batch size lowers
    throughput and batch 64 exhausts an 80 GB GPU (ADR-0009).
    """

    model_id = "AI4Industry/MoonViT-V2"
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        self.model = (
            AutoModel.from_pretrained(self.model_id, dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.num_layers = self.model.config.num_hidden_layers
        self.projector = load_projector(dtype).to(device)

    def preprocess(self) -> Preprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return [round(self.num_layers * (k + 1) / n) for k in range(n)]

    @torch.inference_mode()
    def extract(self, inputs: dict[str, torch.Tensor], image_ids: list[str]) -> Iterator[FeatureBatch]:
        points = self.depth_points()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture(layer: int):
            def hook(_module, _args, output):
                captured[layer] = output.detach()

            return hook

        for layer in points:
            # The last point is the Tower's own output, so it comes off the encoder
            # and carries the final norm. Earlier points come off block layer - 1.
            target = self.model.encoder if layer == self.num_layers else self.model.encoder.blocks[layer - 1]
            handles.append(target.register_forward_hook(capture(layer)))

        try:
            merged = self.model(
                inputs["pixel_values"].to(self.device, self.dtype),
                inputs["grid_thws"].to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        for layer in points:
            yield self._batch(captured[layer].view(rows, -1, self.model.config.hidden_size), image_ids, "tower", layer)

        # Merging runs once, after the last block, so `merged` and `projected` exist at
        # that depth only. Each 2x2 patch group becomes one token of 4 x hidden_size.
        stacked = torch.stack(merged).flatten(2)
        yield self._batch(stacked, image_ids, "merged", self.num_layers)
        yield self._batch(self.projector.post_norm(self.projector.proj(stacked)), image_ids, "projected", self.num_layers)

    def _batch(self, tokens: torch.Tensor, image_ids: list[str], stage: str, layer: int) -> FeatureBatch:
        return FeatureBatch(
            tokens=tokens.cpu(),
            image_ids=image_ids,
            model_id=self.model_id,
            stage=stage,
            layer_index=layer,
            num_layers=self.num_layers,
            resolution=self.resolution,
        )
