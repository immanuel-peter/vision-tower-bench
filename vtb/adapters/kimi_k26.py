from collections.abc import Iterator
from functools import partial
from json import loads
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, PatchPreprocess, default_device, depth_points, make_batch, pack_patches
from vtb.feature_batch import FeatureBatch
from vtb.provenance import log_loaded

MODEL_ID = "immanuelpeter/MoonViT-K2.6"
REVISION = "84ec70ac59184bfa74fca77af9e1cdeccb32b5d6"
PROJECTOR_FILE = "projector.safetensors"
PROJECTOR_CONFIG = "projector_config.json"

# Kimi K2.6 is where scripts/export_moonvit_k26_vision.py reads both halves from.
SOURCE_REPO = "moonshotai/Kimi-K2.6"
SOURCE_SHARDS = ("model-00063-of-000064.safetensors", "model-00064-of-000064.safetensors")
TOWER_PREFIX = "vision_tower."
PROJECTOR_PREFIX = "mm_projector."
PATCH_SIZE = 14


Preprocess = partial(PatchPreprocess, mean=HALF, std=HALF, patch_size=PATCH_SIZE, flatten=False)
collate = partial(pack_patches, grid_key="grid_thws")


def load_projector_weights(dtype: torch.dtype) -> nn.Module:
    """Build Kimi K2.6's patch merger from its published weights."""
    settings = loads(Path(hf_hub_download(MODEL_ID, PROJECTOR_CONFIG, revision=REVISION)).read_text())
    projector = nn.Module()
    projector.pre_norm = nn.LayerNorm(settings["input_size"], eps=settings["norm_eps"])
    projector.proj = nn.Sequential(
        nn.Linear(settings["hidden_size"], settings["hidden_size"]),
        nn.GELU(),
        nn.Linear(settings["hidden_size"], settings["output_size"]),
    )
    projector.load_state_dict(load_file(hf_hub_download(MODEL_ID, PROJECTOR_FILE, revision=REVISION)))
    return projector.to(dtype).eval()


def load_parts(dtype: torch.dtype, attention: str = "eager"):
    log_loaded(MODEL_ID, f"{MODEL_ID}@{REVISION}")
    tower = AutoModel.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype, trust_remote_code=True, attn_implementation=attention
    ).eval()
    return tower, load_projector_weights(dtype)


class KimiK26Adapter:
    """Use batch size 1 without flash attention, as measured."""

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

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

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
        width = self.model.config.hidden_size
        for layer in points:
            yield make_batch(self, captured[layer].view(rows, -1, width), image_ids, "tower", layer)

        stacked = torch.stack(merged)
        yield make_batch(self, stacked.flatten(2), image_ids, "merged", self.num_layers)
        grouped = self.projector.pre_norm(stacked).flatten(2)
        yield make_batch(self, self.projector.proj(grouped), image_ids, "projected", self.num_layers)
