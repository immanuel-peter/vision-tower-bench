"""GLM-5.3 Flash: the published Tower against the pinned parent's tensors and a Tower rebuilt from them."""

import torch
from safetensors.torch import load_file

from vtb.adapters.glm5 import MODEL_ID, REVISION, SOURCE_REPO, SOURCE_REVISION, TOWER_TENSORS, VISION_PREFIX, VISION_SHARD, GLM5Adapter, load_source_tower
from vtb.parity import bit_exact, release_against_source, weights_match
from vtb.provenance import open_release
from vtb.shards import load_prefixed


def validate(device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    release = open_release(MODEL_ID, REVISION, SOURCE_REPO, SOURCE_REVISION)
    parent = load_prefixed(SOURCE_REPO, [VISION_SHARD], VISION_PREFIX, revision=SOURCE_REVISION)
    report = {"weights": {"tower": weights_match(load_file(release / "model.safetensors"), parent, TOWER_TENSORS, extra_ok=True)}}
    report["forward"] = release_against_source(GLM5Adapter(device=device, dtype=dtype), {"model": load_source_tower(dtype)})
    return {**report, "passed": bit_exact(report)}
