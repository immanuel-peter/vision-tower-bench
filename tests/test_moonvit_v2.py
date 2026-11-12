import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from vtb.adapters.moonvit_v2 import KimiK31ProAdapter, MoonViTV2Adapter, collate
from vtb.images import SUFFIXES

RESOLUTION = 448
# The suite runs on an M4 Max and on rented CUDA instances, and the image set on hand
# differs between them, so neither is baked in.
IMAGE_DIR = Path(os.environ.get("VTB_TEST_IMAGES", "data/val2017"))
IMAGES = sorted(p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in SUFFIXES)[:3]
if len(IMAGES) < 3:
    pytest.skip(f"needs three images under {IMAGE_DIR}", allow_module_level=True)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run(adapter_class):
    root = os.environ.get("VTB_KIMI_VALIDATION_ROOT")
    if root:
        bundle = "k3-bundle" if adapter_class is MoonViTV2Adapter else "bundle"
        adapter_class = type("LocalKimiAdapter", (adapter_class,), {"model_id": str(Path(root) / bundle)})
    adapter = adapter_class(resolution=RESOLUTION, device=device())
    prep = adapter.preprocess()
    samples = []
    for path in IMAGES:
        with Image.open(path) as img:
            samples.append(prep(img.convert("RGB")))
    ids = [p.stem for p in IMAGES]
    batches = {}
    for sample, image_id in zip(samples, ids, strict=True):
        for batch in adapter.extract(collate([sample]), [image_id]):
            key = batch.stage + str(batch.layer_index)
            if key in batches:
                previous = batches[key]
                batches[key] = replace(previous, tokens=torch.cat([previous.tokens, batch.tokens]), image_ids=previous.image_ids + batch.image_ids)
            else:
                batches[key] = batch
    return adapter, batches


@pytest.fixture(scope="module", params=[MoonViTV2Adapter, KimiK31ProAdapter])
def extracted(request):
    if os.environ.get("VTB_SKIP_WEIGHTS"):
        pytest.skip("weight tests disabled")
    return run(request.param)


def test_depth_points_cover_the_encoder(extracted):
    ADAPTER, BATCHES = extracted
    assert ADAPTER.num_layers == 27
    assert ADAPTER.depth_points() == [3, 7, 10, 14, 17, 20, 24, 27]


def test_tower_slices_unpack_into_one_row_per_image(extracted):
    ADAPTER, BATCHES = extracted
    for layer in ADAPTER.depth_points():
        tokens = BATCHES[f"tower{layer}"].tokens
        assert tokens.shape == (len(IMAGES), 1024, 1024)


def test_merged_is_a_lossless_regrouping_of_the_last_tower_slice(extracted):
    ADAPTER, BATCHES = extracted
    tower = BATCHES["tower27"].tokens
    merged = BATCHES["merged27"].tokens
    assert merged.shape == (len(IMAGES), 256, 4096)

    rows, _, dim = tower.shape
    regrouped = tower.view(rows, 16, 2, 16, 2, dim).permute(0, 1, 3, 2, 4, 5).reshape(rows, 256, 4 * dim)
    assert torch.equal(regrouped, merged)


def test_projected_maps_merged_into_the_language_model_width(extracted):
    ADAPTER, BATCHES = extracted
    projected = BATCHES["projected27"].tokens
    assert projected.shape == (len(IMAGES), 256, 7168)
    assert ADAPTER.stages == ("tower", "merged", "projected")


def test_pooling_keeps_the_grid_square(extracted):
    ADAPTER, BATCHES = extracted
    assert BATCHES["tower27"].pooled(4).tokens.shape == (len(IMAGES), 16, 1024)
    assert BATCHES["merged27"].pooled(4).tokens.shape == (len(IMAGES), 16, 4096)
