from types import SimpleNamespace

import pytest

from scripts.tower_throughput import configured_patch_size, parser


def adapter_with_patch_size(value):
    return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(patch_size=value)))


def test_configured_patch_size_accepts_scalar_and_square_pair():
    assert configured_patch_size(adapter_with_patch_size(14)) == 14
    assert configured_patch_size(adapter_with_patch_size((16, 16))) == 16


def test_configured_patch_size_rejects_rectangular_patches():
    with pytest.raises(ValueError, match="non-square"):
        configured_patch_size(adapter_with_patch_size((14, 16)))


def test_throughput_default_measures_768_images():
    args = parser().parse_args([
        "--model", "dinov2", "--images", "images", "--out", "result.json",
        "--batch-size", "16",
    ])

    assert args.count == 768
