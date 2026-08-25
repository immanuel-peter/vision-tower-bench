from pathlib import Path

import torch
from PIL import Image

from vtb.adapters.moonvit_v2 import MoonViTV2Adapter, collate

RESOLUTION = 448
IMAGES = sorted(Path("data/val2017").glob("*.jpg"))[:3]


def run():
    adapter = MoonViTV2Adapter(resolution=RESOLUTION, device="mps")
    prep = adapter.preprocess()
    samples = []
    for path in IMAGES:
        with Image.open(path) as img:
            samples.append(prep(img.convert("RGB")))
    ids = [p.stem for p in IMAGES]
    return adapter, {b.stage + str(b.layer_index): b for b in adapter.extract(collate(samples), ids)}


ADAPTER, BATCHES = run()


def test_depth_points_cover_the_encoder():
    assert ADAPTER.num_layers == 27
    assert ADAPTER.depth_points() == [3, 7, 10, 14, 17, 20, 24, 27]


def test_tower_slices_unpack_into_one_row_per_image():
    for layer in ADAPTER.depth_points():
        tokens = BATCHES[f"tower{layer}"].tokens
        assert tokens.shape == (len(IMAGES), 1024, 1024)


def test_merged_is_a_lossless_regrouping_of_the_last_tower_slice():
    tower = BATCHES["tower27"].tokens
    merged = BATCHES["merged27"].tokens
    assert merged.shape == (len(IMAGES), 256, 4096)

    rows, _, dim = tower.shape
    regrouped = tower.view(rows, 16, 2, 16, 2, dim).permute(0, 1, 3, 2, 4, 5).reshape(rows, 256, 4 * dim)
    assert torch.equal(regrouped, merged)


def test_projected_maps_merged_into_the_language_model_width():
    projected = BATCHES["projected27"].tokens
    assert projected.shape == (len(IMAGES), 256, 7168)
    assert ADAPTER.stages == ("tower", "merged", "projected")


def test_pooling_keeps_the_grid_square():
    assert BATCHES["tower27"].pooled(4).tokens.shape == (len(IMAGES), 16, 1024)
    assert BATCHES["merged27"].pooled(4).tokens.shape == (len(IMAGES), 16, 4096)
