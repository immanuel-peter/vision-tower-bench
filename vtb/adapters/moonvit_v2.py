from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from transformers import AutoModel

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop

PATCH_SIZE = 14

MODEL_ID = "immanuelpeter/MoonViT-V2"
PROJECTOR_FILE = "projector.safetensors"

# Kimi K3 is where scripts/export_moonvit_v2.py reads both halves from.
SOURCE_REPO = "moonshotai/Kimi-K3"
PROJECTOR_SHARD = "model-00095-of-000096.safetensors"
PROJECTOR_PREFIX = "mm_projector."
TOWER_SHARD = "model-00096-of-000096.safetensors"
TOWER_PREFIX = "vision_tower."


@dataclass(frozen=True)
class Preprocess:
    resolution: int
    pipeline: transforms.Compose = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pipeline",
            transforms.Compose([
                square_crop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )

    def __call__(self, image) -> torch.Tensor:
        pixels = self.pipeline(image)
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
    """Build Kimi K3's patch merger from its published weights."""
    weights = load_file(hf_hub_download(MODEL_ID, PROJECTOR_FILE))
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
    """Kimi K3 adapter; use batch size 1 without flash attention, as measured in ADR-0009."""

    model_id = MODEL_ID
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
            # Use the encoder output at the final layer to retain its norm.
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

        # Merge each 2x2 patch group.
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
