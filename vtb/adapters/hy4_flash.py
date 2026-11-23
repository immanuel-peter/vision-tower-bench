"""Hy4 flash: a SigLIP2-So400m-derived Tower with a learned 2x2 conv Projector, on a fixed 512 px square."""

import json
import os
import shutil
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from vtb.adapters.base import Inputs
from vtb.adapters.common import (
    HALF,
    ImagePreprocess,
    default_device,
    depth_points,
    make_batch,
    stack_pixels,
)
from vtb.bundle import write_json
from vtb.bundle_manifest import copy_assets
from vtb.feature_batch import FeatureBatch
from vtb.models.hy4_flash import Hy4FlashVisionConfig, Hy4FlashVisionModel
from vtb.provenance import log_loaded, open_release
from vtb.shards import load_prefixed

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "immanuelpeter/Hy4-Flash-Vision"
SOURCE_REPO = "tencent/Hy4-flash"
SOURCE_REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
RELEASE_DIR = ROOT / "hf" / MODEL_ID.split("/")[-1]
MODELING_SOURCE = ROOT / "vtb/models/hy4_flash.py"
MODELING_FILE = "modeling_hy4_flash_vision.py"
PREFIX = "vit."
PROJECTOR_PREFIX = "perceive."
TOWER_TENSORS, PROJECTOR_TENSORS = 435, 6
RESOLUTION, PATCH_SIZE, MERGE_SIZE = 512, 16, 2
# The parent's vision_config fields the standalone model hard-codes, and the values it assumes.
PARENT_LAYOUT = {
    "add_patchemb_bias": True,
    "cat_extra_token": 0,
    "hidden_act": "gelu",
    "interpolate_mode": "bilinear",
    "perceive_post_norm": False,
    "perceive_pre_norm": True,
    "projector_type": "rmsnorm_mlp_adapter",
    "remove_prenorm": True,
    "spatial_merge_size": MERGE_SIZE,
    "patch_size": PATCH_SIZE,
}
CONFIG_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_channels",
    "patch_size",
    "rms_norm_eps",
    "max_image_size",
    "spatial_merge_size",
    "out_hidden_size",
    "hidden_act",
)

Preprocess = partial(
    ImagePreprocess, mean=HALF, std=HALF, rgb=True, multiple=PATCH_SIZE * MERGE_SIZE
)
collate = stack_pixels


def source_json(filename: str) -> dict:
    return json.loads(
        Path(
            hf_hub_download(SOURCE_REPO, filename, revision=SOURCE_REVISION)
        ).read_text()
    )


def source_shards() -> list[str]:
    index = source_json("model.safetensors.index.json")["weight_map"]
    return sorted({shard for name, shard in index.items() if name.startswith(PREFIX)})


def source_weights() -> dict[str, torch.Tensor]:
    """Range-read every ``vit.*`` tensor from the pinned parent, with the prefix removed."""
    index = source_json("model.safetensors.index.json")["weight_map"]
    selected = {name for name in index if name.startswith(PREFIX)}
    weights = load_prefixed(
        SOURCE_REPO, source_shards(), PREFIX, revision=SOURCE_REVISION
    )
    if {PREFIX + name for name in weights} != selected:
        raise ValueError(f"incomplete {SOURCE_REPO} vision state: {len(weights)}")
    projector = sum(name.startswith(PROJECTOR_PREFIX) for name in weights)
    if (len(weights) - projector, projector) != (TOWER_TENSORS, PROJECTOR_TENSORS):
        raise ValueError(
            f"expected {TOWER_TENSORS} Tower and {PROJECTOR_TENSORS} Projector tensors, got {len(weights) - projector} and {projector}"
        )
    return weights


def source_config() -> Hy4FlashVisionConfig:
    vision = source_json("config.json")["vision_config"]
    drift = {
        key: vision.get(key)
        for key, value in PARENT_LAYOUT.items()
        if vision.get(key) != value
    }
    if drift:
        raise ValueError(
            f"parent vision_config differs from the standalone model: {drift}"
        )
    return Hy4FlashVisionConfig(
        **{key: vision[key] for key in CONFIG_FIELDS},
        layer_norm_eps=vision["rms_norm_eps"],
    )


def build(
    config: Hy4FlashVisionConfig, weights: dict[str, torch.Tensor], dtype: torch.dtype
) -> Hy4FlashVisionModel:
    with torch.device("meta"):
        model = Hy4FlashVisionModel(config)
    model.load_state_dict(weights, strict=True, assign=True)
    return model.to(dtype).eval()


def parent_model(dtype: torch.dtype) -> Hy4FlashVisionModel:
    """The standalone model rebuilt from the pinned parent's tensors, without the release."""
    log_loaded(MODEL_ID, f"parent checkpoint {SOURCE_REPO}@{SOURCE_REVISION}")
    return build(source_config(), source_weights(), dtype)


def bundle_override() -> Path | None:
    override = os.environ.get("VTB_HY4_FLASH_BUNDLE")
    return Path(override) if override else None


def open_bundle(bundle: Path | None = None) -> Path:
    return open_release(
        MODEL_ID, SOURCE_REPO, SOURCE_REVISION, bundle=bundle or bundle_override()
    )


def load_model(dtype: torch.dtype, bundle: Path | None = None) -> Hy4FlashVisionModel:
    bundle = open_bundle(bundle)
    config = Hy4FlashVisionConfig.from_pretrained(bundle, local_files_only=True)
    return build(config, load_file(bundle / "model.safetensors"), dtype)


def export(out: Path) -> None:
    weights = source_weights()
    config = source_config()
    build(config, weights, torch.bfloat16)  # Strict names and shapes before writing.
    out.mkdir(parents=True, exist_ok=True)
    copy_assets(RELEASE_DIR, out)
    save_file(weights, out / "model.safetensors", metadata={"format": "pt"})
    shutil.copyfile(MODELING_SOURCE, out / MODELING_FILE)
    shutil.copyfile(
        hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION),
        out / "LICENSE",
    )
    config.architectures = ["Hy4FlashVisionModel"]
    config.dtype = torch.bfloat16
    config.auto_map = {
        "AutoConfig": f"{MODELING_FILE[:-3]}.Hy4FlashVisionConfig",
        "AutoModel": f"{MODELING_FILE[:-3]}.Hy4FlashVisionModel",
    }
    config.save_pretrained(out)
    dtypes: dict[str, int] = {}
    for tensor in weights.values():
        name = str(tensor.dtype).removeprefix("torch.")
        dtypes[name] = dtypes.get(name, 0) + 1
    write_json(
        out / "export_manifest.json",
        {
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_prefix": PREFIX,
            "source_shards": source_shards(),
            "tower_tensors": TOWER_TENSORS,
            "projector_tensors": PROJECTOR_TENSORS,
            "dtype": "bfloat16" if set(dtypes) == {"bfloat16"} else "mixed",
            "tensor_dtypes": dtypes,
            "bytes": sum(tensor.nbytes for tensor in weights.values()),
            "vision_reference": "https://github.com/huggingface/transformers/pull/50903 at c2ec850510a37978ea83df226589623d5f2be776",
        },
    )


class Hy4FlashAdapter:
    model_id = MODEL_ID
    default_resolution = RESOLUTION
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(
        self,
        resolution: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        *,
        bundle: Path | None = None,
    ):
        self.resolution = self.default_resolution if resolution is None else resolution
        if self.resolution != RESOLUTION:
            raise ValueError(
                f"Hy4 flash runs on a fixed {RESOLUTION} px square, a 32x32 patch grid"
            )
        self.device = default_device() if device is None else device
        self.dtype = dtype
        self.model = load_model(dtype, bundle).to(self.device)
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> ImagePreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        pixels = inputs["pixel_values"]
        side = self.resolution
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, side, side):
            raise ValueError(f"expected pixels of shape (batch, 3, {side}, {side})")
        if len(pixels) != len(image_ids) or not len(image_ids):
            raise ValueError("pixel/image ID count differs or batch is empty")
        grid = (side // PATCH_SIZE, side // PATCH_SIZE)
        points = set(self.depth_points())
        hidden = self.model.embeddings(pixels.to(self.device, self.dtype))
        for layer, block in enumerate(self.model.layers, start=1):
            hidden = block(hidden)
            if layer in points:
                yield make_batch(self, hidden, image_ids, "tower", layer)
        # No final norm: the last block feeds the Projector, whose RMSNorm output proj.0 merges 2x2.
        perceive = self.model.perceive
        normed = perceive.norm(hidden)
        merged = perceive.merge(normed, grid)
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        # The newline column the parent appends per row is dropped to keep the grid square.
        projected = perceive.project(normed, grid).flatten(2).transpose(1, 2)
        yield make_batch(self, projected, image_ids, "projected", self.num_layers)
