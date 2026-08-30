from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torchvision import transforms
from transformers import AutoConfig, Qwen3_5VisionModel

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.shards import load_prefixed

MODEL_ID = "Qwen/Qwen3.8-27B"
VISION_SHARD = "model-00001-of-00018.safetensors"
VISION_PREFIX = "model.visual."
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


@dataclass(frozen=True)
class Preprocess:
    """Flatten an image into the patch rows Qwen's tower reads.

    Rows are grouped by 2x2 merge block rather than by raster row, which is the order
    the merger expects. The adapter puts the `tower` Stage back into raster order.
    """

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
        # A still image repeats to fill the temporal patch.
        frames = pixels.unsqueeze(0).expand(TEMPORAL_PATCH_SIZE, -1, -1, -1)
        patches = frames.reshape(
            1, TEMPORAL_PATCH_SIZE, channels,
            rows // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
            cols // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return patches.reshape(rows * cols, channels * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    side = int((samples[0].shape[0]) ** 0.5)
    return {
        "hidden_states": torch.cat(samples),
        "grid_thw": torch.tensor([[1, side, side]] * len(samples)),
    }


def raster(tokens: torch.Tensor, rows: int) -> torch.Tensor:
    """Undo the merge-block grouping so a Stage reshapes to a square grid."""
    width = tokens.shape[-1]
    side = int((tokens.numel() // (rows * width)) ** 0.5)
    blocks = side // MERGE_SIZE
    grid = tokens.view(rows, blocks, blocks, MERGE_SIZE, MERGE_SIZE, width)
    return grid.permute(0, 1, 3, 2, 4, 5).reshape(rows, side * side, width)


def load_tower(dtype: torch.dtype) -> Qwen3_5VisionModel:
    config = AutoConfig.from_pretrained(MODEL_ID).vision_config
    model = Qwen3_5VisionModel._from_config(config, dtype=dtype)
    model.load_state_dict(load_prefixed(MODEL_ID, [VISION_SHARD], VISION_PREFIX))
    return model.eval()


class Qwen3_5Adapter:
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
                captured[layer] = output.detach()

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

        deepest = out.last_hidden_state.view(rows, -1, width * MERGE_SIZE**2)
        yield self._batch(deepest, image_ids, "merged", self.num_layers)
        yield self._batch(out.pooler_output.view(rows, deepest.shape[1], -1), image_ids, "projected", self.num_layers)

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
