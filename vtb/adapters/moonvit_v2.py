from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
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

PATCH_SIZE = 14

MODEL_ID = "immanuelpeter/MoonViT-V2"
PROJECTOR_FILE = "projector.safetensors"

# Kimi K3 is where scripts/export_moonvit_v2_vision.py reads both halves from.
SOURCE_REPO = "moonshotai/Kimi-K3"
PROJECTOR_SHARD = "model-00095-of-000096.safetensors"
PROJECTOR_PREFIX = "mm_projector."
TOWER_SHARD = "model-00096-of-000096.safetensors"
TOWER_PREFIX = "vision_tower."


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    model_id: str
    bundle: str
    bundle_revision: str


K3 = Checkpoint(SOURCE_REPO, "f831ab66814297da540d832a5235f8e904f29d06", MODEL_ID, "MoonViT-V2",
                "bcaedfccd1f07d23ab20de8a0a168668c27cacb3")
K31_PRO = Checkpoint("moonshotai/Kimi-K3.1-Pro", "6d0a0202956796543364d666ab19c6ad1a494ee1",
                     "immanuelpeter/MoonViT-K3.1-Pro", "MoonViT-K3.1-Pro",
                     "6ab95e2daf86e8100be2e712a7a71c0fba26dac4")
CHECKPOINTS = {"moonvit_v2": K3, "kimi_k31_pro": K31_PRO}


Preprocess = partial(PatchPreprocess, mean=HALF, std=HALF, patch_size=PATCH_SIZE, flatten=False)
collate = partial(pack_patches, grid_key="grid_thws")


def load_projector_weights(dtype: torch.dtype, model_id: str = MODEL_ID, revision: str = K3.bundle_revision) -> nn.Module:
    """Build Kimi K3's patch merger from its published weights."""
    local = Path(model_id) / PROJECTOR_FILE
    weights = load_file(local if local.is_file() else hf_hub_download(model_id, PROJECTOR_FILE, revision=revision))
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
    """Kimi K3 adapter; use batch size 1 without flash attention, as measured."""

    checkpoint = K3
    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        log_loaded(self.model_id, f"{self.model_id}@{self.checkpoint.bundle_revision}")
        self.model = (
            AutoModel.from_pretrained(self.model_id, dtype=dtype, revision=self.checkpoint.bundle_revision, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        self.num_layers = self.model.config.num_hidden_layers
        self.projector = load_projector_weights(dtype, self.model_id, self.checkpoint.bundle_revision).to(self.device)

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        if len(image_ids) != 1:
            raise ValueError("Kimi extraction requires batch size 1 for reproducible attention")
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
            yield make_batch(self, captured[layer].view(rows, -1, self.model.config.hidden_size), image_ids, "tower", layer)

        # Merge each 2x2 patch group.
        stacked = torch.stack(merged).flatten(2)
        yield make_batch(self, stacked, image_ids, "merged", self.num_layers)
        yield make_batch(self, self.projector.post_norm(self.projector.proj(stacked)), image_ids, "projected", self.num_layers)


class KimiK31ProAdapter(MoonViTV2Adapter):
    checkpoint = K31_PRO
    model_id = K31_PRO.model_id
