"""Verify that standalone MoonViT-V2 weights match the Tower embedded in Kimi K3."""

import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel

from scripts.export_moonvit_v2 import PROJECTOR_TENSORS, TOWER_TENSORS, projector_config, read_prefix
from vtb.adapters.moonvit_v2 import PROJECTOR_PREFIX, PROJECTOR_SHARD, TOWER_PREFIX, TOWER_SHARD, CHECKPOINTS, MoonViTV2Adapter


@pytest.mark.parametrize("checkpoint", CHECKPOINTS.values())
def test_standalone_tower_matches_the_tower_inside_kimi_k3(checkpoint):
    inside = read_prefix(TOWER_SHARD, TOWER_PREFIX, checkpoint)

    model_id = checkpoint.model_id
    root = os.environ.get("VTB_KIMI_VALIDATION_ROOT")
    if root:
        model_id = str(Path(root) / ("k3-bundle" if checkpoint == CHECKPOINTS["moonvit_v2"] else "bundle"))
    standalone = AutoModel.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True
    ).state_dict()
    # Ignore buffers that are rebuilt at load time.
    standalone = {k: v for k, v in standalone.items() if k in inside}

    assert len(inside) == TOWER_TENSORS
    assert set(standalone) == set(inside)
    for name, weight in standalone.items():
        assert torch.equal(weight, inside[name]), name


@pytest.mark.parametrize("checkpoint", CHECKPOINTS.values())
def test_exported_projector_matches_the_shapes_the_adapter_builds(checkpoint):
    projector = read_prefix(PROJECTOR_SHARD, PROJECTOR_PREFIX, checkpoint)
    assert len(projector) == PROJECTOR_TENSORS

    config = projector_config(projector)
    assert config["input_size"] == config["hidden_size"]
    assert (config["output_size"], config["input_size"]) == tuple(projector["proj.2.weight"].shape)
