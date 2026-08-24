"""Checks that moving image preparation into the adapters kept it correct.

DINOv2 must produce exactly what the old shared pipeline produced. MoonViT-V2 must
produce exactly what its own published processor produces.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor

from vtb.adapters import dinov2, moonvit_v2

RESOLUTION = 448
IMAGES = sorted(Path("data/val2017").glob("*.jpg"))[:4]


def load(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def test_dinov2_matches_the_old_shared_pipeline():
    old = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(dinov2.IMAGENET_MEAN, dinov2.IMAGENET_STD),
    ])
    new = dinov2.Preprocess(RESOLUTION)
    for path in IMAGES:
        image = load(path)
        assert torch.equal(new(image), old(image))


def test_moonvit_matches_its_published_processor():
    processor = AutoImageProcessor.from_pretrained(
        moonvit_v2.MoonViTV2Adapter.model_id, trust_remote_code=True
    )
    crop = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RESOLUTION),
    ])
    ours = moonvit_v2.Preprocess(RESOLUTION)
    for path in IMAGES:
        image = load(path)
        reference = processor(crop(image), return_tensors="pt")
        mine = ours(image)
        assert mine.shape == reference["pixel_values"].shape
        assert torch.allclose(mine, reference["pixel_values"], atol=1e-6)
        assert torch.equal(
            moonvit_v2.collate([mine])["grid_thws"], reference["grid_thws"]
        )
