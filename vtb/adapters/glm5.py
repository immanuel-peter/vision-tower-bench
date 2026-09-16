from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torchvision import transforms
from transformers import AutoConfig, AutoModel, Glm5NextVisionModel

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/GLM-5.3-Flash-Vision"
BUNDLE = Path(__file__).resolve().parents[2] / "hf/GLM-5.3-Flash-Vision"

SOURCE_REPO = "zai-org/GLM-5.3-Flash"
SOURCE_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
VISION_SHARD = "model-00062-of-00062.safetensors"
VISION_PREFIX = "model.visual."
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class Preprocess:
    """Flatten an image into merge-block patch rows. ``raster`` undoes that grouping."""

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
        "hidden_states": torch.cat(samples),
        "grid_thw": torch.tensor([[1, side, side]] * len(samples)),
    }


def raster(tokens: torch.Tensor, rows: int) -> torch.Tensor:
    width = tokens.shape[-1]
    side = int((tokens.numel() // (rows * width)) ** 0.5)
    blocks = side // MERGE_SIZE
    grid = tokens.view(rows, blocks, blocks, MERGE_SIZE, MERGE_SIZE, width)
    return grid.permute(0, 1, 3, 2, 4, 5).reshape(rows, side * side, width)


def load_source_tower(dtype: torch.dtype = torch.bfloat16) -> Glm5NextVisionModel:
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = Glm5NextVisionModel._from_config(config.vision_config, dtype=dtype)
    weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], VISION_PREFIX, revision=SOURCE_REVISION
    )
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in weights.items()})
    return tower.eval()


def load_tower(dtype: torch.dtype) -> Glm5NextVisionModel:
    if (BUNDLE / "model.safetensors").exists():
        return AutoModel.from_pretrained(BUNDLE, dtype=dtype).eval()
    return load_source_tower(dtype)


class GLM5Adapter:
    model_id = MODEL_ID
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        self.model = load_tower(dtype).to(device)
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.num_layers = self.model.config.depth

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
            handles.append(self.model.blocks[layer - 1].register_forward_hook(capture(layer)))

        try:
            out = self.model(
                inputs["hidden_states"].to(self.device, self.dtype),
                inputs["grid_thw"].to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size
        for layer in points:
            tokens = captured[layer].view(rows, -1, width)
            yield self._batch(raster(tokens, rows), image_ids, "tower", layer)

        merged = out.last_hidden_state.view(rows, -1, self.model.config.out_hidden_size)
        yield self._batch(merged, image_ids, "merged", self.num_layers)
        yield self._batch(
            out.pooler_output.view(rows, merged.shape[1], -1), image_ids, "projected", self.num_layers
        )

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
