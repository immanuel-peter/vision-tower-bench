from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def square_crop(resolution: int):
    """Resize the short side and center crop, giving every Tower the same framing.

    Framing is shared because it decides what content reaches the Tower. Everything
    downstream of it, normalization and packing, belongs to the adapter.
    """
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
    ])


class ImageFolder(Dataset):
    """Reads images and hands each one to the adapter's transform.

    The transform must be picklable and must not hold the model, because DataLoader
    workers run it in their own processes.
    """

    def __init__(self, root: Path, transform: Callable[[Image.Image], Any], limit: int | None = None):
        paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no images under {root}")
        self.paths = paths[:limit]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Any, str]:
        path = self.paths[i]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), path.stem
