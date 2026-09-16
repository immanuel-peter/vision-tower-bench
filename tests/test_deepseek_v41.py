from PIL import Image

from vtb.adapters.deepseek_v41 import DOWNSAMPLE, PATCH_SIZE, Preprocess


def test_448_is_a_32_by_32_patch_grid():
    patches = Preprocess(448)(Image.new("RGB", (600, 480), "gray"))
    side = int(patches.shape[0] ** 0.5)
    assert side == 32
    assert patches.shape == (side * side, 3, PATCH_SIZE, PATCH_SIZE)
    padded = side + (-side % DOWNSAMPLE)
    assert padded // DOWNSAMPLE == 11
