import torch
from PIL import Image

from vtb.adapters.gemma4 import PATCH_SIZE, POOL_SIZE, Preprocess, padded_side, patchify


def test_448_pads_to_a_30_by_30_grid():
    assert padded_side(448) == 480
    assert padded_side(480) == 480
    patches = Preprocess(448)(Image.new("RGB", (600, 480), "gray"))
    side = int(patches.shape[0] ** 0.5)
    assert side == 30
    assert side % POOL_SIZE == 0
    assert patches.shape == (side * side, 3 * PATCH_SIZE * PATCH_SIZE)


def test_patchify_is_row_major_with_channel_last():
    pixels = torch.arange(2 * 32 * 32, dtype=torch.float32).view(2, 32, 32)
    patches = patchify(pixels)
    assert patches.shape == (4, 2 * 16 * 16)
    # First patch is the top-left 16x16, channels stacked last.
    assert patches[0, 0].item() == pixels[0, 0, 0].item()
    assert patches[1, 0].item() == pixels[0, 0, 16].item()
