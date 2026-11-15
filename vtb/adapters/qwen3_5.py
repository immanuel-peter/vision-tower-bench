import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import torch
from transformers import AutoModel, Qwen3_5VisionConfig, Qwen3_5VisionModel

from vtb.shards import load_prefixed

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, PatchPreprocess, default_device, depth_points, make_batch, pack_patches, raster
from vtb.feature_batch import FeatureBatch
from vtb.provenance import check_provenance, log_loaded

ROOT = Path(__file__).resolve().parents[2]
VISION_PREFIX = "model.visual."
TOWER_TENSORS = 333


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str
    out_hidden_size: int
    license: str
    release_revision: str

    @property
    def bundle(self) -> Path:
        override = os.environ.get(f"VTB_{self.name.upper()}_BUNDLE")
        return Path(override) if override else ROOT / "hf" / self.model_id.split("/")[-1]


QWEN27B = Checkpoint(
    "qwen3_5", "immanuelpeter/Qwen3.8-27B-Vision", "Qwen/Qwen3.8-27B",
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", 5120, "apache-2.0", "60a1bf9336197d19458290fd43f3d527f6cf3078",
)
FLASH_NEXT = Checkpoint(
    "qwen38_flash_next", "immanuelpeter/Qwen3.8-Flash-Next-Vision", "Qwen/Qwen3.8-Flash-Next",
    "de4b8e4d43b917e7706784d8bb445c9af86a3540", 2560, "qwen-community-1.0", "120ed5d5baf11cb478099350e94218ee3cbed3df",
)
CHECKPOINTS = {c.name: c for c in (QWEN27B, FLASH_NEXT)}
MODEL_ID, SOURCE_REPO = QWEN27B.model_id, QWEN27B.repo


def source_json(checkpoint: Checkpoint, filename: str) -> dict:
    return json.loads(Path(hf_hub_download(checkpoint.repo, filename, revision=checkpoint.revision)).read_text())


def source_weights(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    selected = {k: v for k, v in index.items() if k.startswith(VISION_PREFIX)}
    weights = load_prefixed(checkpoint.repo, sorted(set(selected.values())), VISION_PREFIX, revision=checkpoint.revision)
    if len(weights) != TOWER_TENSORS or {VISION_PREFIX + k for k in weights} != set(selected):
        raise ValueError(f"incomplete {checkpoint.name} vision state: {len(weights)} tensors")
    return weights


def load_source_tower(checkpoint: Checkpoint, dtype=torch.bfloat16) -> Qwen3_5VisionModel:
    config = source_json(checkpoint, "config.json")["vision_config"]
    config.update(model_type="qwen3_5_vision", out_hidden_size=checkpoint.out_hidden_size)
    model = Qwen3_5VisionModel._from_config(Qwen3_5VisionConfig(**config), dtype=dtype)
    model.load_state_dict(source_weights(checkpoint), strict=True)
    return model.eval()


PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


Preprocess = partial(PatchPreprocess, mean=HALF, std=HALF, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE, temporal_patch_size=TEMPORAL_PATCH_SIZE)
collate = partial(pack_patches, key="hidden_states")


def load_tower(dtype: torch.dtype, checkpoint: Checkpoint = QWEN27B) -> Qwen3_5VisionModel:
    if (checkpoint.bundle / "model.safetensors").is_file():
        check_provenance(checkpoint.bundle, checkpoint.repo, checkpoint.revision)
        log_loaded(checkpoint.model_id, str(checkpoint.bundle))
        model = AutoModel.from_pretrained(checkpoint.bundle, dtype=dtype, local_files_only=True).eval()
    else:
        log_loaded(checkpoint.model_id, f"{checkpoint.model_id}@{checkpoint.release_revision}")
        model = AutoModel.from_pretrained(
            checkpoint.model_id, revision=checkpoint.release_revision, dtype=dtype,
        ).eval()
    if model.config.out_hidden_size != checkpoint.out_hidden_size:
        raise ValueError(f"wrong Projector width for {checkpoint.name}")
    return model


class QwenAdapter:
    checkpoint = QWEN27B
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        Preprocess(self.resolution)
        self.model = load_tower(dtype, self.checkpoint).to(self.device)
        self.model_id = self.checkpoint.model_id
        self.num_layers = self.model.config.depth

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
            yield make_batch(self, raster(tokens, rows), image_ids, "tower", layer)

        deepest = out.last_hidden_state.view(rows, -1, width * MERGE_SIZE**2)
        yield make_batch(self, deepest, image_ids, "merged", self.num_layers)
        yield make_batch(self, out.pooler_output.view(rows, deepest.shape[1], -1), image_ids, "projected", self.num_layers)


class Qwen3_5Adapter(QwenAdapter):
    checkpoint = QWEN27B
    model_id = QWEN27B.model_id


class Qwen38FlashNextAdapter(QwenAdapter):
    checkpoint = FLASH_NEXT
    model_id = FLASH_NEXT.model_id
