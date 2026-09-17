from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from transformers import AutoConfig, AutoModel

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/C-RADIOv4-H"
BUNDLE = Path(__file__).resolve().parents[2] / "hf/C-RADIOv4-H"

SOURCE_REPO = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
SOURCE_REVISION = "e5e9932441de940c9a62185c870ea5bcd4cd24e2"
VISION_SHARD = "model-00001-of-00017.safetensors"
TOWER_PREFIX = "vision_model."
PROJECTOR_PREFIX = "mlp1."
PATCH_SIZE = 16
DOWNSAMPLE = 0.5
VIT_WIDTH = 1280
PROJECTOR_HIDDEN = 20480
LLM_WIDTH = 2688


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * hidden_states).to(dtype)


def pixel_shuffle(tokens: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    # InternVL v2 shuffle used by Nemotron Omni extract_feature.
    batch, width, height, channels = tokens.size()
    tokens = tokens.view(batch, width, int(height * scale_factor), int(channels / scale_factor))
    tokens = tokens.permute(0, 2, 1, 3).contiguous()
    tokens = tokens.view(
        batch,
        int(height * scale_factor),
        int(width * scale_factor),
        int(channels / (scale_factor * scale_factor)),
    )
    return tokens.permute(0, 2, 1, 3).contiguous()


def make_projector() -> nn.Sequential:
    merged = VIT_WIDTH * int(1 / DOWNSAMPLE) ** 2
    return nn.Sequential(
        RMSNorm(merged, eps=1e-5),
        nn.Linear(merged, PROJECTOR_HIDDEN, bias=False),
        SquaredReLU(),
        nn.Linear(PROJECTOR_HIDDEN, LLM_WIDTH, bias=False),
    )


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
            ]),
        )

    def __call__(self, image) -> torch.Tensor:
        return self.pipeline(image)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.stack(samples)}


def _cast_radio(radio, dtype: torch.dtype):
    radio = radio.to(dtype)
    conditioner = radio.radio_model.input_conditioner
    if hasattr(conditioner, "dtype"):
        conditioner.dtype = dtype
    return radio


def attach_video_embedder(radio) -> None:
    generator = radio.radio_model.model.patch_generator
    if hasattr(generator, "video_embedder"):
        return
    generator.video_embedder = nn.Linear(
        2 * 3 * generator.patch_size * generator.patch_size,
        generator.embed_dim,
        bias=False,
    )


def load_source_parts(dtype: torch.dtype = torch.bfloat16):
    config = AutoConfig.from_pretrained(SOURCE_REPO, trust_remote_code=True, revision=SOURCE_REVISION)
    radio = AutoModel.from_config(config.vision_config, trust_remote_code=True)
    attach_video_embedder(radio)
    weights = load_prefixed(SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION)
    radio.load_state_dict({name: tensor.to(dtype) for name, tensor in weights.items()}, strict=False)
    radio = _cast_radio(radio, dtype)
    projector = make_projector()
    projector.load_state_dict({
        name: tensor.to(dtype)
        for name, tensor in load_prefixed(
            SOURCE_REPO, [VISION_SHARD], PROJECTOR_PREFIX, revision=SOURCE_REVISION
        ).items()
    })
    return radio.eval(), projector.to(dtype).eval()


def load_parts(dtype: torch.dtype):
    if (BUNDLE / "model.safetensors").exists():
        config = AutoConfig.from_pretrained(SOURCE_REPO, trust_remote_code=True, revision=SOURCE_REVISION)
        radio = AutoModel.from_config(config.vision_config, trust_remote_code=True)
        attach_video_embedder(radio)
        radio.load_state_dict({
            name: tensor.to(dtype) for name, tensor in load_file(BUNDLE / "model.safetensors").items()
        }, strict=False)
        radio = _cast_radio(radio, dtype)
        projector = make_projector()
        projector.load_state_dict({
            name: tensor.to(dtype) for name, tensor in load_file(BUNDLE / "projector.safetensors").items()
        })
        return radio.eval(), projector.to(dtype).eval()
    return load_source_parts(dtype)


class NemotronOmniAdapter:
    model_id = MODEL_ID
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        radio, projector = load_parts(dtype)
        self.model = radio.to(device)
        self.projector = projector.to(device)
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.num_layers = len(self.model.radio_model.model.blocks)

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
                tokens = output[0] if isinstance(output, tuple) else output
                captured[layer] = tokens.detach()

            return hook

        blocks = self.model.radio_model.model.blocks
        for layer in points:
            handles.append(blocks[layer - 1].register_forward_hook(capture(layer)))

        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        try:
            out = self.model(pixels)
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        height = pixels.shape[-2] // PATCH_SIZE
        width = pixels.shape[-1] // PATCH_SIZE
        skip = self.model.radio_model.model.patch_generator.num_skip

        def spatial(tokens: torch.Tensor) -> torch.Tensor:
            if tokens.ndim == 3:
                tokens = tokens[:, skip:, :]
            return tokens.reshape(rows, height * width, -1)

        for layer in points:
            yield self._batch(spatial(captured[layer]), image_ids, "tower", layer)

        features = out.features
        tower = features.reshape(rows, height, width, -1)
        merged = pixel_shuffle(tower, DOWNSAMPLE)
        merged = merged.reshape(rows, -1, merged.shape[-1])
        yield self._batch(merged, image_ids, "merged", self.num_layers)
        yield self._batch(self.projector(merged), image_ids, "projected", self.num_layers)

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
