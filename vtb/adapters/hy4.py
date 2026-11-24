"""Tencent Hy4 and Hy4 flash Towers with their learned 2x2 conv Projector, on a fixed 512 px square."""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vtb.adapters.base import Inputs
from vtb.adapters.common import (
    HALF,
    ImagePreprocess,
    default_device,
    depth_points,
    make_batch,
    stack_pixels,
)
from vtb.bundle import ROOT, write_bundle
from vtb.bundle_manifest import copy_assets, write_manifest
from vtb.feature_batch import FeatureBatch
from vtb.models.hy4 import (
    MERGE,
    Hy4FlashVisionConfig,
    Hy4FlashVisionModel,
    Hy4VisionConfig,
    Hy4VisionModel,
)
from vtb.provenance import open_release
from vtb.shards import load_prefixed

MODELING_SOURCE = ROOT / "vtb/models/hy4.py"
PREFIX = "vit."
PROJECTOR_PREFIX = "perceive."
RESOLUTION, PATCH_SIZE = 512, 16
# The parent's vision_config fields the standalone models hard-code, and the values they assume.
PARENT_LAYOUT = {
    "add_patchemb_bias": True,
    "cat_extra_token": 0,
    "hidden_act": "gelu",
    "interpolate_mode": "bilinear",
    "num_channels": 3,
    "perceive_post_norm": False,
    "perceive_pre_norm": True,
    "projector_type": "rmsnorm_mlp_adapter",
    "remove_prenorm": True,
    "spatial_merge_size": MERGE,
    "patch_size": PATCH_SIZE,
}
CONFIG_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "patch_size",
    "rms_norm_eps",
    "max_image_size",
    "out_hidden_size",
)


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str
    tower_tensors: int
    projector_tensors: int
    config_class: type[Hy4VisionConfig]
    model_class: type
    modeling_file: str
    vision_reference: str

    @property
    def release_dir(self) -> Path:
        return ROOT / "hf" / self.model_id.split("/")[-1]

    def open(self, bundle: Path | None = None) -> Path:
        override = os.environ.get(f"VTB_{self.name.upper()}_BUNDLE")
        return open_release(
            self.model_id,
            self.repo,
            self.revision,
            bundle=bundle or (Path(override) if override else None),
        )


HY4 = Checkpoint(
    "hy4",
    "immanuelpeter/Hy4-Vision",
    "tencent/Hy4",
    "c10c0de5b6f850703fb0a90e65d1c32b558b9167",
    643,
    6,
    Hy4VisionConfig,
    Hy4VisionModel,
    "modeling_hy4.py",
    "https://github.com/huggingface/transformers/commit/1031277cef60200fda98628e9adc8fb15f0e1549",
)
HY4_FLASH = Checkpoint(
    "hy4_flash",
    "immanuelpeter/Hy4-Flash-Vision",
    "tencent/Hy4-flash",
    "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959",
    435,
    6,
    Hy4FlashVisionConfig,
    Hy4FlashVisionModel,
    "modeling_hy4_flash_vision.py",
    "https://github.com/huggingface/transformers/pull/50903 at c2ec850510a37978ea83df226589623d5f2be776",
)
CHECKPOINTS = {item.name: item for item in (HY4, HY4_FLASH)}

Preprocess = partial(
    ImagePreprocess, mean=HALF, std=HALF, rgb=True, multiple=PATCH_SIZE * MERGE
)
collate = stack_pixels


def source_file(checkpoint: Checkpoint, filename: str) -> Path:
    return Path(
        hf_hub_download(checkpoint.repo, filename, revision=checkpoint.revision)
    )


def source_json(checkpoint: Checkpoint, filename: str) -> dict:
    return json.loads(source_file(checkpoint, filename).read_text())


def source_shards(checkpoint: Checkpoint) -> list[str]:
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    return sorted({shard for name, shard in index.items() if name.startswith(PREFIX)})


def source_weights(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    """Range-read every ``vit.*`` tensor from the pinned parent, with the prefix removed."""
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    selected = {name for name in index if name.startswith(PREFIX)}
    weights = load_prefixed(
        checkpoint.repo,
        source_shards(checkpoint),
        PREFIX,
        revision=checkpoint.revision,
    )
    if {PREFIX + name for name in weights} != selected:
        raise ValueError(f"incomplete {checkpoint.repo} vision state: {len(weights)}")
    projector = sum(name.startswith(PROJECTOR_PREFIX) for name in weights)
    expected = (checkpoint.tower_tensors, checkpoint.projector_tensors)
    if (len(weights) - projector, projector) != expected:
        raise ValueError(
            f"expected {expected[0]} Tower and {expected[1]} Projector tensors, got {len(weights) - projector} and {projector}"
        )
    return weights


def source_config(checkpoint: Checkpoint) -> Hy4VisionConfig:
    vision = source_json(checkpoint, "config.json")["vision_config"]
    drift = {
        key: vision.get(key)
        for key, value in PARENT_LAYOUT.items()
        if vision.get(key) != value
    }
    if drift:
        raise ValueError(
            f"parent vision_config differs from the standalone model: {drift}"
        )
    return checkpoint.config_class(**{key: vision[key] for key in CONFIG_FIELDS})


def build(
    checkpoint: Checkpoint,
    config: Hy4VisionConfig,
    weights: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> torch.nn.Module:
    with torch.device("meta"):
        model = checkpoint.model_class(config)
    model.load_state_dict(weights, strict=True, assign=True)
    return model.to(dtype).eval()


def load_model(
    checkpoint: Checkpoint, dtype: torch.dtype, bundle: Path | None = None
) -> torch.nn.Module:
    bundle = checkpoint.open(bundle)
    config = checkpoint.config_class.from_pretrained(bundle, local_files_only=True)
    return build(checkpoint, config, load_file(bundle / "model.safetensors"), dtype)


def export(out: Path, checkpoint: Checkpoint) -> None:
    weights = source_weights(checkpoint)
    config = source_config(checkpoint)
    build(
        checkpoint, config, weights, torch.bfloat16
    )  # Strict names and shapes before writing.
    stem = checkpoint.modeling_file.removesuffix(".py")
    config.architectures = [checkpoint.model_class.__name__]
    config.dtype = torch.bfloat16
    config.auto_map = {
        "AutoConfig": f"{stem}.{checkpoint.config_class.__name__}",
        "AutoModel": f"{stem}.{checkpoint.model_class.__name__}",
    }
    write_bundle(
        out,
        weights={"model.safetensors": weights},
        license=source_file(checkpoint, "LICENSE"),
        files={checkpoint.modeling_file: MODELING_SOURCE},
    )
    config.save_pretrained(out)
    copy_assets(checkpoint.release_dir, out)
    write_manifest(
        out,
        checkpoint.repo,
        checkpoint.revision,
        source_shards(checkpoint),
        merger_in_model=True,
        projector_prefix=PROJECTOR_PREFIX,
        extra={
            "source_prefix": PREFIX,
            "bytes": sum(tensor.nbytes for tensor in weights.values()),
            "vision_reference": checkpoint.vision_reference,
        },
    )


class Hy4TowerAdapter:
    checkpoint: Checkpoint
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
                f"{self.checkpoint.name} runs on a fixed {RESOLUTION} px square, a {RESOLUTION // PATCH_SIZE}x{RESOLUTION // PATCH_SIZE} patch grid"
            )
        self.device = default_device() if device is None else device
        self.dtype = dtype
        self.model_id = self.checkpoint.model_id
        self.model = load_model(self.checkpoint, dtype, bundle).to(self.device)
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
        hidden, merged, projected = self.model.stages(
            pixels.to(self.device, self.dtype), self.depth_points()
        )
        for layer, tokens in hidden.items():
            yield make_batch(self, tokens, image_ids, "tower", layer)
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        yield make_batch(self, projected, image_ids, "projected", self.num_layers)


class Hy4Adapter(Hy4TowerAdapter):
    checkpoint = HY4


class Hy4FlashAdapter(Hy4TowerAdapter):
    checkpoint = HY4_FLASH
