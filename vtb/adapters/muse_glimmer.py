from collections.abc import Iterator
from dataclasses import dataclass
from json import loads
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, MuseGlimmerVisionModel
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerRMSNorm,
    MuseGlimmerVisionAdapter,
)
from transformers.vision_utils import get_vision_window_index

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, ImagePreprocess, default_device, depth_points, make_batch, pack_patches
from vtb.feature_batch import FeatureBatch
from vtb.provenance import log_loaded

MODEL_ID = "immanuelpeter/Muse-Glimmer-Vision"
REVISION = "c0cd7ffcb881966e04c4adead0f057f8ceb09447"
PROJECTOR_FILE = "projector.safetensors"
PROJECTOR_CONFIG = "projector_config.json"

# Muse Glimmer 30B is where scripts/export_muse_glimmer_vision.py reads both halves from.
SOURCE_REPO = "meta-models/Muse-Glimmer-30B"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
TOWER_PREFIX = "model.vision_tower."
ADAPTER_PREFIX = "model.vision_adapter."
PROJECTION_PREFIX = "model.vision_projection."
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


@dataclass(frozen=True)
class Preprocess(ImagePreprocess):
    mean: tuple[float, ...] | None = HALF
    std: tuple[float, ...] | None = HALF

    def __call__(self, image) -> torch.Tensor:
        pixels = self.pixels(image).unsqueeze(0)
        _, channels, height, width = pixels.shape
        rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
        patches = pixels.view(1, channels, rows, PATCH_SIZE, cols, PATCH_SIZE)
        # Each flattened patch is laid out (temporal, channel), not (channel, temporal).
        patches = patches.permute(0, 2, 4, 1, 3, 5).unsqueeze(3)
        patches = patches.expand(-1, -1, -1, TEMPORAL_PATCH_SIZE, -1, -1, -1)
        return patches.reshape(rows * cols, TEMPORAL_PATCH_SIZE * channels * PATCH_SIZE * PATCH_SIZE)


collate = pack_patches


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter = MuseGlimmerVisionAdapter(config)
        self.projection = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=False)
        self.norm = MuseGlimmerRMSNorm(eps=config.text_config.rms_norm_eps, with_scale=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(self.adapter(tokens)))


def projector_shapes(settings: dict) -> SimpleNamespace:
    return SimpleNamespace(
        out_hidden_size=settings["input_size"],
        projector_hidden_size=settings["hidden_size"],
        projector_hidden_act=settings["activation_func"],
        text_config=SimpleNamespace(
            hidden_size=settings["output_size"],
            rms_norm_eps=settings["norm_eps"],
        ),
    )


def load_parts(dtype: torch.dtype) -> tuple[MuseGlimmerVisionModel, Projector]:
    log_loaded(MODEL_ID, f"{MODEL_ID}@{REVISION}")
    tower = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, dtype=dtype).eval()
    settings = loads(Path(hf_hub_download(MODEL_ID, PROJECTOR_CONFIG, revision=REVISION)).read_text())
    projector = Projector(projector_shapes(settings))
    projector.load_state_dict(load_file(hf_hub_download(MODEL_ID, PROJECTOR_FILE, revision=REVISION)))
    return tower, projector.to(dtype).eval()


class MuseGlimmerAdapter:
    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        tower, projector = load_parts(dtype)
        self.model = tower.to(self.device)
        self.projector = projector.to(self.device)
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> Preprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    def _unwindow(self, grid_thw: torch.Tensor) -> torch.Tensor:
        config = self.model.config
        window_index, _ = get_vision_window_index(
            grid_thw,
            spatial_merge_size=1,
            window_size=config.pos_emb_height * config.patch_size,
            patch_size=config.patch_size,
        )
        return torch.argsort(window_index)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
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
            yield make_batch(self, tokens.view(rows, -1, width), image_ids, "tower", layer)

        # The tower already merges each 2x2 block, so its output is the merged Stage.
        merged = out.last_hidden_state.view(rows, -1, width * MERGE_SIZE**2)
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        yield make_batch(self, self.projector(merged), image_ids, "projected", self.num_layers)
