"""Forward parity: standalone MoonViT-V2 matches the Tower inside Kimi K3."""

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel

from scripts.export_moonvit_v2 import PROJECTOR_TENSORS, TOWER_TENSORS, read_prefix
from vtb.adapters import moonvit_v2
from vtb.adapters.moonvit_v2 import (
    PROJECTOR_PREFIX,
    PROJECTOR_SHARD,
    TOWER_PREFIX,
    TOWER_SHARD,
    MoonViTV2Adapter,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


def _build_projector(weights: dict[str, torch.Tensor], dtype: torch.dtype) -> nn.Module:
    """Rebuild Kimi K3's patch merger from explicitly given weights."""
    width, merged_width = weights["proj.2.weight"].shape
    projector = nn.Module()
    projector.proj = nn.Sequential(
        nn.Linear(merged_width, merged_width, bias=False),
        nn.GELU(),
        nn.Linear(merged_width, width, bias=False),
    )
    projector.post_norm = nn.RMSNorm(width, eps=1e-5)
    projector.load_state_dict(weights)
    return projector.to(dtype).eval()


def test_standalone_forward_matches_the_tower_inside_kimi_k3():
    tower = {k: v.to(DTYPE) for k, v in read_prefix(TOWER_SHARD, TOWER_PREFIX).items()}
    assert len(tower) == TOWER_TENSORS
    projector_weights = {k: v.to(DTYPE) for k, v in read_prefix(PROJECTOR_SHARD, PROJECTOR_PREFIX).items()}
    assert len(projector_weights) == PROJECTOR_TENSORS

    adapter = MoonViTV2Adapter(resolution=448, dtype=DTYPE, device=DEVICE.type)
    parent = AutoModel.from_pretrained(
        MoonViTV2Adapter.model_id, dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)
    incompatible = parent.load_state_dict(tower, strict=False)
    assert not incompatible.unexpected_keys
    parent.eval()
    parent_projector = _build_projector(projector_weights, DTYPE).to(DEVICE.type)

    sample = moonvit_v2.Preprocess(448)(Image.new("RGB", (640, 480), "gray"))
    inputs = moonvit_v2.collate([sample])

    with torch.inference_mode():
        pixel_values = inputs["pixel_values"].to(DEVICE, DTYPE)
        grid_thws = inputs["grid_thws"].to(DEVICE)
        actual_merged = adapter.model(pixel_values, grid_thws)
        expected_merged = parent(pixel_values, grid_thws)
        actual_stacked = torch.stack(actual_merged).flatten(2)
        expected_stacked = torch.stack(expected_merged).flatten(2)
        actual_projected = adapter.projector.post_norm(adapter.projector.proj(actual_stacked))
        expected_projected = parent_projector.post_norm(parent_projector.proj(expected_stacked))

    for actual, expected in zip(actual_merged, expected_merged, strict=True):
        assert torch.equal(actual, expected)
    assert torch.equal(actual_projected, expected_projected)
