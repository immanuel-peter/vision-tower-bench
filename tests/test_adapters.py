"""These tests download about 10 GB of weights across the roster. Set
VTB_SKIP_WEIGHTS=1 to skip them and keep the rest of the suite fast.
"""

import os

import pytest
import torch
from PIL import Image

from vtb.adapters.dinov2 import DINOv2Adapter
from vtb.adapters.kimi_k26 import KimiK26Adapter
from vtb.adapters.moonvit_v2 import MoonViTV2Adapter
from vtb.adapters.muse_glimmer import MuseGlimmerAdapter
from vtb.adapters.qwen3_5 import MERGE_SIZE, PATCH_SIZE, Qwen3_5Adapter, raster
from vtb.adapters.siglip2 import SigLIP2Adapter
from vtb.extract import ADAPTERS

RESOLUTION = 448

# Tower grid side, merged grid side, and the widths each Stage should produce.
EXPECTED = {
    DINOv2Adapter: (32, None, 1024, None, None),
    SigLIP2Adapter: (32, None, 1152, None, None),
    MoonViTV2Adapter: (32, 16, 1024, 4096, 7168),
    KimiK26Adapter: (32, 16, 1152, 4608, 7168),
    Qwen3_5Adapter: (28, 14, 1152, 4608, 5120),
    MuseGlimmerAdapter: (32, 16, 1536, 6144, 6656),
}

pytestmark = pytest.mark.skipif(
    os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1"
)


@pytest.fixture(scope="module")
def image():
    return Image.new("RGB", (600, 480), "gray")


def test_every_roster_adapter_is_registered_for_extraction():
    assert set(ADAPTERS.values()) == set(EXPECTED)


@pytest.mark.parametrize("adapter_class", EXPECTED, ids=lambda c: c.__name__)
def test_adapter_yields_every_stage_at_the_expected_shape(adapter_class, image):
    tower_side, merged_side, tower_width, merged_width, projected_width = EXPECTED[adapter_class]
    adapter = adapter_class(resolution=RESOLUTION, device="cpu", dtype=torch.float32)
    batch = adapter.collate([adapter.preprocess()(image)])
    yielded = list(adapter.extract(batch, ["one"]))

    assert {b.stage for b in yielded} == set(adapter.stages)
    towers = [b for b in yielded if b.stage == "tower"]
    assert [b.layer_index for b in towers] == adapter.depth_points()
    assert towers[-1].relative_depth == 1.0

    for batch_out in towers:
        assert tuple(batch_out.tokens.shape) == (1, tower_side**2, tower_width)

    for stage, side, width in (("merged", merged_side, merged_width), ("projected", merged_side, projected_width)):
        if width is None:
            assert stage not in adapter.stages
            continue
        only = [b for b in yielded if b.stage == stage]
        assert len(only) == 1
        assert tuple(only[0].tokens.shape) == (1, side**2, width)


# Projectors group 2x2 grid squares, then flatten by position or channel.
MERGE_LAYOUT = {
    MoonViTV2Adapter: "position",
    KimiK26Adapter: "position",
    Qwen3_5Adapter: "position",
    MuseGlimmerAdapter: "channel",
}


@pytest.mark.parametrize("adapter_class", MERGE_LAYOUT, ids=lambda c: c.__name__)
def test_merging_only_regroups_tower_tokens(adapter_class, image):
    adapter = adapter_class(resolution=RESOLUTION, device="cpu", dtype=torch.bfloat16)
    batch = adapter.collate([adapter.preprocess()(image)])
    yielded = {(b.stage, b.layer_index): b for b in adapter.extract(batch, ["one"])}

    tower = yielded[("tower", adapter.num_layers)].tokens
    merged = yielded[("merged", adapter.num_layers)].tokens
    rows, count, width = tower.shape
    side = int(count**0.5)
    blocks = tower.view(rows, side // 2, 2, side // 2, 2, width)
    blocks = blocks.permute(0, 1, 3, 2, 4, 5).reshape(rows, -1, 4, width)
    if MERGE_LAYOUT[adapter_class] == "channel":
        blocks = blocks.permute(0, 1, 3, 2)

    assert torch.equal(blocks.reshape(merged.shape), merged)
    assert not torch.equal(tower.reshape(merged.shape), merged)


def test_qwen_preprocess_groups_patches_by_merge_block():
    side = RESOLUTION // PATCH_SIZE
    ids = torch.arange(side * side, dtype=torch.float32).view(1, side, side)
    pixels = ids.repeat_interleave(PATCH_SIZE, 1).repeat_interleave(PATCH_SIZE, 2)

    blocks = side // MERGE_SIZE
    grouped = pixels.reshape(1, 1, 1, blocks, MERGE_SIZE, PATCH_SIZE, blocks, MERGE_SIZE, PATCH_SIZE)
    grouped = grouped.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(side * side, PATCH_SIZE * PATCH_SIZE)

    assert grouped[:4, 0].tolist() == [0.0, 1.0, float(side), float(side + 1)]
    restored = raster(grouped[:, :1].unsqueeze(0), 1)
    assert torch.equal(restored.flatten(), ids.flatten())
