"""count-v1 feeds pre-cropped squares. Every adapter's preprocessing must then
reduce to one plain bicubic square resize: no second crop, no shift."""

import sys

import numpy as np
import pytest
import torch
from PIL import Image

from vtb.extract import ADAPTERS

RESOLUTION = {"dinov3": 512}


def square(side: int) -> Image.Image:
    rng = np.random.default_rng(side)
    y, x = np.mgrid[:side, :side]
    pattern = np.stack([x % 256, y % 256, (x // 5 + y // 3) % 256], -1) + rng.integers(0, 40, (side, side, 3))
    return Image.fromarray(pattern.clip(0, 255).astype("uint8"))


def as_tensors(value):
    if isinstance(value, dict):
        return value
    return {"value": value}


@pytest.mark.parametrize("model", sorted(ADAPTERS))
@pytest.mark.parametrize("side", [301, 384, 448, 960])
def test_square_input_reduces_to_a_plain_bicubic_resize(model, side):
    resolution = RESOLUTION.get(model, 448)
    preprocess = sys.modules[ADAPTERS[model].__module__].Preprocess(resolution)
    image = square(side)
    resized = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    actual, expected = as_tensors(preprocess(image)), as_tensors(preprocess(resized))
    for key in expected:
        assert torch.equal(actual[key], expected[key]), (model, side, key)


@pytest.mark.parametrize("model", sorted(ADAPTERS))
def test_the_only_geometric_steps_are_the_shared_resize_and_centre_crop(model):
    from torchvision import transforms

    resolution = RESOLUTION.get(model, 448)
    pipeline = sys.modules[ADAPTERS[model].__module__].Preprocess(resolution).pipeline.transforms
    crop, *rest = pipeline
    resize, centre = crop.transforms
    assert isinstance(resize, transforms.Resize) and resize.size == resolution
    assert resize.interpolation == transforms.InterpolationMode.BICUBIC
    assert isinstance(centre, transforms.CenterCrop) and centre.size == (resolution, resolution)
    assert [type(step).__name__ for step in rest] in (["ToTensor"], ["ToTensor", "Normalize"])
