"""Extract raster-ordered features from the Tencent Hy4 Tower and Projector."""

import json
import os
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vtb.adapters.common import (
    HALF,
    ImagePreprocess,
    default_device,
    depth_points,
    make_batch,
    stack_pixels,
)
from vtb.bundle import ROOT, write_bundle, write_json
from vtb.bundle_manifest import copy_assets
from vtb.models.hy4 import Hy4VisionConfig, Hy4VisionModel
from vtb.provenance import open_release
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/Hy4-Vision"
SOURCE_REPO = "tencent/Hy4"
SOURCE_REVISION = "c10c0de5b6f850703fb0a90e65d1c32b558b9167"
VISUAL_TENSORS = 649
RESOLUTION = 512
RELEASE = ROOT / "hf" / "Hy4-Vision"
Preprocess = partial(ImagePreprocess, mean=HALF, std=HALF, rgb=True)


def source_file(filename):
    return Path(hf_hub_download(SOURCE_REPO, filename, revision=SOURCE_REVISION))


def source_weights():
    index = json.loads(source_file("model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    selected = {
        name.removeprefix("vit."): shard
        for name, shard in index.items()
        if name.startswith("vit.")
    }
    shards = sorted(set(selected.values()))
    if len(selected) != VISUAL_TENSORS or len(shards) != 8:
        raise ValueError("unexpected Hy4 vision tensor or shard count")
    weights = load_prefixed(SOURCE_REPO, shards, "vit.", revision=SOURCE_REVISION)
    if weights.keys() != selected.keys() or any(
        t.dtype != torch.bfloat16 for t in weights.values()
    ):
        raise ValueError("Hy4 weights do not match the pinned BF16 vision index")
    return weights


def build_model(config, weights):
    with torch.device("meta"):
        model = Hy4VisionModel(config)
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval()


def export(out: Path):
    weights = source_weights()
    config = Hy4VisionConfig(
        architectures=["Hy4VisionModel"],
        auto_map={
            "AutoConfig": "modeling_hy4.Hy4VisionConfig",
            "AutoModel": "modeling_hy4.Hy4VisionModel",
        },
        dtype="bfloat16",
    )
    build_model(config, weights)
    write_bundle(
        out,
        weights={"model.safetensors": weights},
        configs={"config.json": config.to_dict()},
        license=source_file("LICENSE"),
        files={
            "modeling_hy4.py": ROOT / "vtb/models/hy4.py",
        },
    )
    copy_assets(RELEASE, out)
    write_json(
        out / "export_manifest.json",
        {
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "tensors": len(weights),
            "bytes": sum(t.nbytes for t in weights.values()),
        },
    )


class Hy4Adapter:
    model_id = MODEL_ID
    default_resolution = RESOLUTION
    stages = ("tower", "merged", "projected")
    collate = staticmethod(stack_pixels)

    def __init__(
        self, resolution=None, dtype=torch.bfloat16, device=None, *, bundle=None
    ):
        self.resolution = self.default_resolution if resolution is None else resolution
        if self.resolution != self.default_resolution:
            raise ValueError(f"Hy4 requires resolution {self.default_resolution}")
        self.dtype = dtype
        self.device = default_device() if device is None else device
        override = os.environ.get("VTB_HY4_BUNDLE")
        release = open_release(
            MODEL_ID,
            SOURCE_REPO,
            SOURCE_REVISION,
            bundle=bundle or (Path(override) if override else None),
        )
        config = Hy4VisionConfig.from_pretrained(release, local_files_only=True)
        self.model = build_model(config, load_file(release / "model.safetensors")).to(
            device=self.device, dtype=dtype
        )
        self.num_layers = config.num_hidden_layers

    def preprocess(self):
        return Preprocess(self.resolution)

    def depth_points(self, n=8):
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs, image_ids):
        pixels = inputs["pixel_values"]
        if (
            pixels.shape != (len(image_ids), 3, self.resolution, self.resolution)
            or not image_ids
        ):
            raise ValueError("Hy4 expects one 3×512×512 image per ID")
        hidden, merged, projected = self.model(
            pixels.to(self.device, self.dtype), self.depth_points()
        )
        for layer, tokens in hidden.items():
            yield make_batch(self, tokens, image_ids, "tower", layer)
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        yield make_batch(self, projected, image_ids, "projected", self.num_layers)
