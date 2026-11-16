"""Shared measurements for checking a release against an independent reference forward."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    a, e = actual.detach().cpu().double().flatten(), expected.detach().cpu().double().flatten()
    delta = a - e
    return {
        "shape": list(actual.shape),
        "bit_exact": torch.equal(actual, expected),
        "max_abs": delta.abs().max().item(),
        "reference_max_abs": e.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / e.norm().clamp_min(1e-12)).item(),
        "cosine": max(-1.0, min(1.0, F.cosine_similarity(a, e, dim=0).item())),
        "finite": bool(torch.isfinite(actual).all()),
    }


def reference_raster(packed: torch.Tensor, grid_thw: torch.Tensor, column_major: bool = False) -> torch.Tensor:
    """Scatter packed 2x2 merge blocks back to raster order by coordinate, independent of adapter reshapes."""
    restored, start = [], 0
    for t, h, w in grid_thw.tolist():
        assert t == 1
        image = torch.empty(h, w, packed.shape[-1], device=packed.device, dtype=packed.dtype)
        outer = [(y, x) for y in range(0, h, 2) for x in range(0, w, 2)]
        if column_major:
            outer.sort(key=lambda coord: (coord[1], coord[0]))
        for y, x in outer:
            for dy in range(2):
                for dx in range(2):
                    image[y + dy, x + dx] = packed[start]
                    start += 1
        restored.append(image.flatten(0, 1))
    assert start == len(packed)
    return torch.stack(restored)
