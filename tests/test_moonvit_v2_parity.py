"""Verify that standalone MoonViT-V2 weights match the Tower embedded in Kimi K3."""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel

from vtb.adapters.moonvit_v2 import PROJECTOR_REPO, MoonViTV2Adapter

TOWER_SHARD = "model-00096-of-000096.safetensors"
TOWER_PREFIX = "vision_tower."


def test_standalone_tower_matches_the_tower_inside_kimi_k3():
    inside = load_file(hf_hub_download(PROJECTOR_REPO, TOWER_SHARD))
    inside = {k.removeprefix(TOWER_PREFIX): v for k, v in inside.items() if k.startswith(TOWER_PREFIX)}

    standalone = AutoModel.from_pretrained(
        MoonViTV2Adapter.model_id, dtype=torch.bfloat16, trust_remote_code=True
    ).state_dict()
    # Ignore buffers that are rebuilt at load time.
    standalone = {k: v for k, v in standalone.items() if k in inside}

    assert len(inside) == 165
    assert set(standalone) == set(inside)
    for name, weight in standalone.items():
        assert torch.equal(weight, inside[name]), name
