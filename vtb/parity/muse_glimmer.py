"""Muse Glimmer: the published Tower and Projector against the pinned parent's tensors and modules rebuilt from them."""

import torch
from safetensors.torch import load_file

from vtb.adapters.muse_glimmer import (
    ADAPTER_PREFIX, MODEL_ID, PROJECTION_PREFIX, PROJECTOR_TENSORS, SHARDS, SOURCE_REPO, SOURCE_REVISION, TOWER_PREFIX, TOWER_TENSORS,
    MuseGlimmerAdapter, load_source_parts,
)
from vtb.parity import bit_exact, release_against_source, weights_match
from vtb.provenance import open_release
from vtb.shards import load_prefixed


def validate(device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    release = open_release(MODEL_ID, SOURCE_REPO, SOURCE_REVISION)
    def parent(prefix):
        return load_prefixed(SOURCE_REPO, SHARDS, prefix, revision=SOURCE_REVISION)
    projector = {f"{module}.{name}": tensor for module, prefix in (("adapter", ADAPTER_PREFIX), ("projection", PROJECTION_PREFIX))
                 for name, tensor in parent(prefix).items()}
    report = {"weights": {
        "tower": weights_match(load_file(release / "model.safetensors"), parent(TOWER_PREFIX), TOWER_TENSORS),
        "projector": weights_match(load_file(release / "projector.safetensors"), projector, PROJECTOR_TENSORS),
    }}
    tower, projector = load_source_parts(dtype)
    report["forward"] = release_against_source(MuseGlimmerAdapter(device=device, dtype=dtype), {"model": tower, "projector": projector})
    return {**report, "passed": bit_exact(report)}
