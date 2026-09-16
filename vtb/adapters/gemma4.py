from collections.abc import Iterator
from dataclasses import dataclass, field
from json import loads
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from transformers import AutoConfig, AutoModel, Gemma4VisionModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/Gemma4-31B-Vision"
BUNDLE = Path(__file__).resolve().parents[2] / "hf/Gemma4-31B-Vision"

# google/gemma-4-31B-it is where scripts/export_gemma4_vision.py reads both halves from.
SOURCE_REPO = "google/gemma-4-31B-it"
SOURCE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
VISION_SHARD = "model-00001-of-00002.safetensors"
TOWER_PREFIX = "model.vision_tower."
PROJECTOR_PREFIX = "model.embed_vision."
PATCH_SIZE = 16
POOL_SIZE = 3
ALIGN = PATCH_SIZE * POOL_SIZE
PAD_VALUE = 0.5


def padded_side(resolution: int) -> int:
    remainder = resolution % ALIGN
    return resolution if remainder == 0 else resolution + (ALIGN - remainder)


def patchify(pixels: torch.Tensor) -> torch.Tensor:
    channels, height, width = pixels.shape
    rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
    patches = pixels.reshape(channels, rows, PATCH_SIZE, cols, PATCH_SIZE)
    return patches.permute(1, 3, 2, 4, 0).reshape(rows * cols, -1)


def position_ids(side: int, rows: int) -> torch.Tensor:
    xs, ys = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="xy")
    grid = torch.stack((xs, ys), dim=-1).reshape(1, side * side, 2)
    return grid.expand(rows, -1, -1).contiguous()


@dataclass(frozen=True)
class Preprocess:
    """Square-crop, then pad bottom-right so the 3x3 pool divides the grid."""

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
        pixels = self.pipeline(image)
        target = padded_side(self.resolution)
        _, height, width = pixels.shape
        pixels = F.pad(pixels, (0, target - width, 0, target - height), value=PAD_VALUE)
        return patchify(pixels)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    side = int(samples[0].shape[0] ** 0.5)
    return {
        "pixel_values": torch.stack(samples),
        "pixel_position_ids": position_ids(side, len(samples)),
    }


class Projector(nn.Module):
    def __init__(self, settings: dict):
        super().__init__()
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            settings["input_size"], eps=settings["norm_eps"], with_scale=False
        )
        self.embedding_projection = nn.Linear(
            settings["input_size"], settings["output_size"], bias=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(tokens))


def projector_settings(vision_config, text_hidden: int) -> dict:
    return {
        "input_size": vision_config.hidden_size,
        "output_size": text_hidden,
        "norm_eps": vision_config.rms_norm_eps,
    }


def load_source_parts(dtype: torch.dtype) -> tuple[Gemma4VisionModel, Projector]:
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = Gemma4VisionModel._from_config(config.vision_config, dtype=dtype)
    weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION
    )
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in weights.items()})
    projector = Projector(projector_settings(config.vision_config, config.text_config.hidden_size))
    projector.load_state_dict({
        name: tensor.to(dtype)
        for name, tensor in load_prefixed(
            SOURCE_REPO, [VISION_SHARD], PROJECTOR_PREFIX, revision=SOURCE_REVISION
        ).items()
    })
    return tower.eval(), projector.to(dtype).eval()


def load_projector(dtype: torch.dtype, bundle: Path) -> Projector:
    settings = loads((bundle / "projector_config.json").read_text())
    projector = Projector(settings)
    projector.load_state_dict(load_file(bundle / "projector.safetensors"))
    return projector.to(dtype).eval()


def load_parts(dtype: torch.dtype) -> tuple[Gemma4VisionModel, Projector]:
    if (BUNDLE / "model.safetensors").exists():
        tower = AutoModel.from_pretrained(BUNDLE, dtype=dtype).eval()
        return tower, load_projector(dtype, BUNDLE)
    return load_source_parts(dtype)


class Gemma4Adapter:
    model_id = MODEL_ID
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        tower, projector = load_parts(dtype)
        self.model = tower.to(device)
        self.projector = projector.to(device)
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.num_layers = self.model.config.num_hidden_layers

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

        for layer in points:
            handles.append(self.model.encoder.layers[layer - 1].register_forward_hook(capture(layer)))

        try:
            out = self.model(
                inputs["pixel_values"].to(self.device, self.dtype),
                inputs["pixel_position_ids"].to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size
        for layer in points:
            yield self._batch(captured[layer].view(rows, -1, width), image_ids, "tower", layer)

        merged = out.last_hidden_state
        if merged.ndim == 2:
            merged = merged.view(rows, -1, width)
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
