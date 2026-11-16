"""Kimi K2.6: the published MoonViT and its shipped code against the pinned parent's tensors and remote code."""

import importlib.util

import torch
from safetensors.torch import load_file
from transformers import AutoModel

from vtb.adapters.kimi_k26 import (
    MODEL_ID, PROJECTOR_PREFIX, PROJECTOR_TENSORS, REVISION, SOURCE_REPO, SOURCE_REVISION, SOURCE_SHARDS, TOWER_PREFIX, TOWER_TENSORS,
    Preprocess, collate, load_source_parts,
)
from vtb.parity import bit_exact, compare, fixed_images, weights_match
from vtb.provenance import open_release
from vtb.shards import load_prefixed


def release_projector(bundle):
    spec = importlib.util.spec_from_file_location("kimi_k26_release_projector", bundle / "projector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_projector(bundle)


def validate(device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    bundle = open_release(MODEL_ID, REVISION, SOURCE_REPO, SOURCE_REVISION)
    report = {"weights": {
        part: weights_match(load_file(bundle / filename), load_prefixed(SOURCE_REPO, SOURCE_SHARDS, prefix, revision=SOURCE_REVISION), count)
        for part, filename, prefix, count in (
            ("tower", "model.safetensors", TOWER_PREFIX, TOWER_TENSORS),
            ("projector", "projector.safetensors", PROJECTOR_PREFIX, PROJECTOR_TENSORS),
        )
    }}
    source, source_projector = (module.to(device) for module in load_source_parts(dtype, attention="eager"))
    released = AutoModel.from_pretrained(bundle, dtype=dtype, local_files_only=True, trust_remote_code=True).to(device).eval()
    projector = release_projector(bundle).to(device, dtype).eval()
    report["forward"] = {}
    for index, image in enumerate(fixed_images()):
        inputs = collate([Preprocess(448)(image)])
        pixels, grid = inputs["pixel_values"].to(device, dtype), inputs["grid_thws"].to(device)
        with torch.inference_mode():
            expected, actual = source(pixels, grid), released(pixels, grid)
            stages = {"merged": (actual[0], expected[0]), "projected": (projector(actual)[0], source_projector(expected)[0])}
        for stage, (got, want) in stages.items():
            report["forward"][f"{index}:{stage}"] = compare(got, want)
    return {**report, "passed": bit_exact(report)}
