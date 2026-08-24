from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(resolution: int):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageFolder(Dataset):
    def __init__(self, root: Path, resolution: int, limit: int | None = None):
        paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no images under {root}")
        self.paths = paths[:limit]
        self.transform = preprocess(resolution)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        path = self.paths[i]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), path.stem
