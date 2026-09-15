import torch
from PIL import Image

from vtb.adapters.minimax_m3 import MERGE_SIZE, PATCH_SIZE, Preprocess, merge_2x2


def test_448_is_a_32_by_32_patch_grid():
    patches = Preprocess(448)(Image.new("RGB", (600, 480), "gray"))
    side = int(patches.shape[0] ** 0.5)
    assert side == 32
    assert side % MERGE_SIZE == 0
    assert patches.shape == (side * side, 3 * 2 * PATCH_SIZE * PATCH_SIZE)


def test_merge_2x2_groups_spatial_neighbors():
    rows, side, width = 1, 4, 3
    tokens = torch.arange(rows * side * side * width, dtype=torch.float32).view(rows, side * side, width)
    merged = merge_2x2(tokens)
    assert merged.shape == (1, 4, 12)
    # Top-left 2x2 of a 4x4 raster grid, flattened by position.
    top_left = tokens[0, [0, 1, 4, 5]].reshape(-1)
    assert torch.equal(merged[0, 0], top_left)
