from PIL import Image

from vtb.adapters.nemotron_omni import PATCH_SIZE, Preprocess


def test_448_is_a_28_by_28_pixel_grid():
    pixels = Preprocess(448)(Image.new("RGB", (600, 480), "gray"))
    assert pixels.shape == (3, 448, 448)
    assert 448 % PATCH_SIZE == 0
    assert (448 // PATCH_SIZE) % 2 == 0
