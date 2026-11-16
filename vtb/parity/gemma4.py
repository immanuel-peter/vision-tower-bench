"""Gemma 4: the published Tower and Projector against the pinned parent's tensors and modules rebuilt from them."""

import torch
from safetensors.torch import load_file

from vtb.adapters.gemma4 import (
    MODEL_ID, PROJECTOR_PREFIX, PROJECTOR_TENSORS, SOURCE_REPO, SOURCE_REVISION, TOWER_PREFIX, TOWER_TENSORS, VISION_SHARD,
    Gemma4Adapter, load_source_parts,
)
from vtb.parity import bit_exact, release_against_source, weights_match
from vtb.provenance import open_release
from vtb.shards import load_prefixed


def validate(device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    release = open_release(MODEL_ID, SOURCE_REPO, SOURCE_REVISION)
    report = {"weights": {
        part: weights_match(load_file(release / filename), load_prefixed(SOURCE_REPO, [VISION_SHARD], prefix, revision=SOURCE_REVISION), count, extra_ok=True)
        for part, filename, prefix, count in (
            ("tower", "model.safetensors", TOWER_PREFIX, TOWER_TENSORS),
            ("projector", "projector.safetensors", PROJECTOR_PREFIX, PROJECTOR_TENSORS),
        )
    }}
    tower, projector = load_source_parts(dtype)
    report["forward"] = release_against_source(Gemma4Adapter(device=device, dtype=dtype), {"model": tower, "projector": projector})
    return {**report, "passed": bit_exact(report)}
