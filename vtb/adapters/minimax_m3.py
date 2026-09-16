from collections.abc import Iterator
from dataclasses import dataclass, field
from json import loads
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLConfig,
    MiniMaxM3VLVisionConfig,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLMultiModalProjector,
    MiniMaxM3VLVisionModel,
)

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/MiniMax-M3-Vision"
BUNDLE = Path(__file__).resolve().parents[2] / "hf/MiniMax-M3-Vision"

SOURCE_REPO = "MiniMaxAI/MiniMax-M3"
SOURCE_REVISION = "f0e1c1e04d40177e4673a22097036854f536e9c0"
VISION_SHARD = "model-00059-of-00059.safetensors"
PROJECTOR_SHARDS = (
    "model-00026-of-00059.safetensors",
    "model-00059-of-00059.safetensors",
)
TOWER_PREFIX = "vision_tower.vision_model."
PROJECTOR_PREFIX = "multi_modal_projector."
MERGE_MLP_PREFIX = "patch_merge_mlp."
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def remap_tower(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for name, tensor in weights.items():
        name = name.replace("encoder.layers.", "layers.")
        name = name.replace("embeddings.patch_embedding.", "embeddings.proj.")
        remapped[name] = tensor
    return remapped


def remap_projector(linear: dict[str, torch.Tensor], merge: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = dict(linear)
    for name, tensor in merge.items():
        remapped[name.replace("linear_", "merge_linear_")] = tensor
    return remapped


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
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]),
        )

    def __call__(self, image) -> torch.Tensor:
        pixels = self.pipeline(image)
        channels, height, width = pixels.shape
        rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
        frames = pixels.unsqueeze(0).expand(TEMPORAL_PATCH_SIZE, -1, -1, -1)
        patches = frames.reshape(
            1, TEMPORAL_PATCH_SIZE, channels,
            rows // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
            cols // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return patches.reshape(rows * cols, channels * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    side = int(samples[0].shape[0] ** 0.5)
    return {
        "pixel_values": torch.cat(samples),
        "grid_thw": torch.tensor([[1, side, side]] * len(samples)),
    }


def raster(tokens: torch.Tensor, rows: int) -> torch.Tensor:
    width = tokens.shape[-1]
    side = int((tokens.numel() // (rows * width)) ** 0.5)
    blocks = side // MERGE_SIZE
    grid = tokens.view(rows, blocks, blocks, MERGE_SIZE, MERGE_SIZE, width)
    return grid.permute(0, 1, 3, 2, 4, 5).reshape(rows, side * side, width)


def merge_2x2(tokens: torch.Tensor) -> torch.Tensor:
    rows, count, width = tokens.shape
    side = int(count ** 0.5)
    blocks = tokens.view(rows, side // MERGE_SIZE, MERGE_SIZE, side // MERGE_SIZE, MERGE_SIZE, width)
    return blocks.permute(0, 1, 3, 2, 4, 5).reshape(rows, -1, MERGE_SIZE * MERGE_SIZE * width)


def load_source_parts(dtype: torch.dtype = torch.bfloat16) -> tuple[MiniMaxM3VLVisionModel, nn.Module]:
    config = MiniMaxM3VLConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = MiniMaxM3VLVisionModel._from_config(config.vision_config, dtype=dtype)
    tower.load_state_dict({
        name: tensor.to(dtype)
        for name, tensor in remap_tower(
            load_prefixed(SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION)
        ).items()
    })
    projector = MiniMaxM3VLMultiModalProjector(config)
    projector.load_state_dict({
        name: tensor.to(dtype)
        for name, tensor in remap_projector(
            load_prefixed(SOURCE_REPO, list(PROJECTOR_SHARDS), PROJECTOR_PREFIX, revision=SOURCE_REVISION),
            load_prefixed(SOURCE_REPO, list(PROJECTOR_SHARDS), MERGE_MLP_PREFIX, revision=SOURCE_REVISION),
        ).items()
    })
    return tower.eval(), projector.to(dtype).eval()


def projector_from_settings(settings: dict, dtype: torch.dtype) -> nn.Module:
    config = MiniMaxM3VLConfig(
        vision_config=MiniMaxM3VLVisionConfig(
            hidden_size=settings["input_size"],
            spatial_merge_size=MERGE_SIZE,
        ),
        text_config={"hidden_size": settings["output_size"]},
        projector_hidden_size=settings["hidden_size"],
    )
    projector = MiniMaxM3VLMultiModalProjector(config)
    return projector.to(dtype).eval()


def load_parts(dtype: torch.dtype) -> tuple[MiniMaxM3VLVisionModel, nn.Module]:
    if (BUNDLE / "model.safetensors").exists():
        config = MiniMaxM3VLVisionConfig.from_pretrained(BUNDLE)
        tower = MiniMaxM3VLVisionModel._from_config(config, dtype=dtype)
        tower.load_state_dict({
            name: tensor.to(dtype) for name, tensor in load_file(BUNDLE / "model.safetensors").items()
        })
        settings = loads((BUNDLE / "projector_config.json").read_text())
        projector = projector_from_settings(settings, dtype)
        projector.load_state_dict({
            name: tensor.to(dtype) for name, tensor in load_file(BUNDLE / "projector.safetensors").items()
        })
        return tower.eval(), projector.to(dtype).eval()
    return load_source_parts(dtype)


class MiniMaxM3Adapter:
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
            handles.append(self.model.layers[layer - 1].register_forward_hook(capture(layer)))

        try:
            out = self.model(
                inputs["pixel_values"].to(self.device, self.dtype),
                inputs["grid_thw"].to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size

        def packed(tokens: torch.Tensor) -> torch.Tensor:
            if tokens.ndim == 3:
                tokens = tokens.reshape(-1, tokens.shape[-1])
            return tokens.view(rows, -1, width)

        for layer in points:
            yield self._batch(raster(packed(captured[layer]), rows), image_ids, "tower", layer)

        last_packed = packed(out.last_hidden_state)
        merged = merge_2x2(raster(last_packed, rows))
        yield self._batch(merged, image_ids, "merged", self.num_layers)
        projected = self.projector(last_packed.reshape(-1, width)).view(rows, merged.shape[1], -1)
        yield self._batch(projected, image_ids, "projected", self.num_layers)

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
