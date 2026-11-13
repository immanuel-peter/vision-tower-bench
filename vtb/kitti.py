from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_DIVISOR = 256.0


def sample_key(path: Path, kind: str) -> str:
    marker = f"_sync_{kind}_"
    if marker not in path.stem:
        raise ValueError(f"{path.name} does not contain {marker}")
    return path.stem.replace(marker, "_sync_", 1)


def paired_samples(root: Path) -> list[tuple[str, Path, Path]]:
    images = {sample_key(path, "image"): path for path in (root / "image").glob("*.png")}
    depths = {
        sample_key(path, "groundtruth_depth"): path
        for path in (root / "groundtruth_depth").glob("*.png")
    }
    if not images or images.keys() != depths.keys():
        missing_images = sorted(depths.keys() - images.keys())[:3]
        missing_depths = sorted(images.keys() - depths.keys())[:3]
        raise ValueError(
            f"unpaired KITTI selection: missing images {missing_images}, missing depths {missing_depths}"
        )
    return [(key, images[key], depths[key]) for key in sorted(images)]


def read_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        encoded = np.array(image, dtype=np.uint16)
    return encoded.astype(np.float32) / DEPTH_DIVISOR
