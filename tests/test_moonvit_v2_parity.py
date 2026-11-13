"""Verify both exported checkpoints against their own pinned parent tensors."""

import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel

from scripts.export_moonvit_v2_vision import PROJECTOR_TENSORS, TOWER_TENSORS, discover_shards, projector_config, read_parent
from vtb.adapters.moonvit_v2 import PROJECTOR_PREFIX, TOWER_PREFIX, CHECKPOINTS

pytestmark = pytest.mark.skipif(os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1")


def parent_tensors(checkpoint, prefix):
    return read_parent(checkpoint, discover_shards(checkpoint)[prefix], prefix)


@pytest.mark.parametrize("checkpoint", CHECKPOINTS.values())
def test_exported_tower_matches_its_pinned_parent(checkpoint):
    inside = parent_tensors(checkpoint, TOWER_PREFIX)

    model_id = checkpoint.model_id
    root = os.environ.get("VTB_KIMI_VALIDATION_ROOT")
    if root:
        model_id = str(Path(root) / ("k3-bundle" if checkpoint == CHECKPOINTS["moonvit_v2"] else "bundle"))
    standalone = AutoModel.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True,
        **({"revision": checkpoint.bundle_revision} if not root and checkpoint.bundle_revision else {})
    ).state_dict()
    # Ignore buffers that are rebuilt at load time.
    standalone = {k: v for k, v in standalone.items() if k in inside}

    assert len(inside) == TOWER_TENSORS
    assert set(standalone) == set(inside)
    for name, weight in standalone.items():
        assert torch.equal(weight, inside[name]), name


@pytest.mark.parametrize("checkpoint", CHECKPOINTS.values())
def test_exported_projector_matches_the_shapes_the_adapter_builds(checkpoint):
    projector = parent_tensors(checkpoint, PROJECTOR_PREFIX)
    assert len(projector) == PROJECTOR_TENSORS

    config = projector_config(projector)
    assert config["input_size"] == config["hidden_size"]
    assert (config["output_size"], config["input_size"]) == tuple(projector["proj.2.weight"].shape)
