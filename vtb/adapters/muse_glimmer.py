from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch import nn
from torchvision import transforms
from transformers import AutoConfig, MuseGlimmerVisionModel
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerRMSNorm,
    MuseGlimmerVisionAdapter,
)
from transformers.vision_utils import get_vision_window_index

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.shards import load_prefixed

MODEL_ID = "meta-models/Muse-Glimmer-30B"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
TOWER_PREFIX = "model.vision_tower."
ADAPTER_PREFIX = "model.vision_adapter."
PROJECTION_PREFIX = "model.vision_projection."
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


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
        pixels = self.pipeline(image).unsqueeze(0)
        _, channels, height, width = pixels.shape
        rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
        patches = pixels.view(1, channels, rows, PATCH_SIZE, cols, PATCH_SIZE)
        # Each flattened patch is laid out (temporal, channel), not (channel, temporal).
        patches = patches.permute(0, 2, 4, 1, 3, 5).unsqueeze(3)
        patches = patches.expand(-1, -1, -1, TEMPORAL_PATCH_SIZE, -1, -1, -1)
        return patches.reshape(rows * cols, TEMPORAL_PATCH_SIZE * channels * PATCH_SIZE * PATCH_SIZE)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    side = int(samples[0].shape[0] ** 0.5)
    return {
        "pixel_values": torch.cat(samples),
        "grid_thw": torch.tensor([[1, side, side]] * len(samples)),
    }


class Projector(nn.Module):
    """Muse Glimmer maps merged tokens to the text width in two learned steps."""

    def __init__(self, config):
        super().__init__()
        self.adapter = MuseGlimmerVisionAdapter(config)
        self.projection = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=False)
        self.norm = MuseGlimmerRMSNorm(eps=config.text_config.rms_norm_eps, with_scale=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(self.adapter(tokens)))


def load_parts(dtype: torch.dtype) -> tuple[MuseGlimmerVisionModel, Projector]:
    config = AutoConfig.from_pretrained(MODEL_ID)
    tower = MuseGlimmerVisionModel._from_config(config.vision_config, dtype=dtype)
    tower.load_state_dict(load_prefixed(MODEL_ID, SHARDS, TOWER_PREFIX))

    projector = Projector(config)
    projector.adapter.load_state_dict(load_prefixed(MODEL_ID, SHARDS, ADAPTER_PREFIX))
    projector.projection.load_state_dict(load_prefixed(MODEL_ID, SHARDS, PROJECTION_PREFIX))
    return tower.eval(), projector.to(dtype).eval()


class MuseGlimmerAdapter:
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

    def _unwindow(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """The tower reorders tokens into attention windows; this reverses that."""
        config = self.model.config
        window_index, _ = get_vision_window_index(
            grid_thw,
            spatial_merge_size=1,
            window_size=config.pos_emb_height * config.patch_size,
            patch_size=config.patch_size,
        )
        return torch.argsort(window_index)

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
            # ln_post runs after the window order is undone, so the deepest point keeps its norm.
            target = self.model.ln_post if layer == self.num_layers else self.model.layers[layer - 1]
            handles.append(target.register_forward_hook(capture(layer)))

        grid_thw = inputs["grid_thw"].to(self.device)
        try:
            out = self.model(inputs["pixel_values"].to(self.device, self.dtype), grid_thw)
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size
        order = self._unwindow(grid_thw)
        for layer in points:
            tokens = captured[layer] if layer == self.num_layers else captured[layer][order]
            yield self._batch(tokens.view(rows, -1, width), image_ids, "tower", layer)

        # The tower already merges each 2x2 block, so its output is the merged Stage.
        merged = out.last_hidden_state.view(rows, -1, width * MERGE_SIZE**2)
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
