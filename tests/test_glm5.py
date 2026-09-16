from PIL import Image

from vtb.adapters.glm5 import MERGE_SIZE, PATCH_SIZE, Preprocess


def test_448_is_a_32_by_32_patch_grid():
    patches = Preprocess(448)(Image.new("RGB", (600, 480), "gray"))
    side = int(patches.shape[0] ** 0.5)
    assert side == 32
    assert side % MERGE_SIZE == 0
    assert patches.shape == (side * side, 3 * 2 * PATCH_SIZE * PATCH_SIZE)
