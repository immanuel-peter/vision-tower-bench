from dataclasses import dataclass, field
from math import isqrt

import torch
from torchvision import transforms

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
HALF = (0.5, 0.5, 0.5)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def depth_points(num_layers: int, n: int = 8) -> list[int]:
    if not 1 <= n <= num_layers:
        raise ValueError(f"depth count must be between 1 and {num_layers}")
    return [round(num_layers * (k + 1) / n) for k in range(n)]


def make_batch(adapter, tokens: torch.Tensor, image_ids: list[str], stage: str, layer: int) -> FeatureBatch:
    return FeatureBatch(
        tokens=tokens.cpu(),
        image_ids=image_ids,
        model_id=adapter.model_id,
        stage=stage,
        layer_index=layer,
        num_layers=adapter.num_layers,
        resolution=adapter.resolution,
    )


def raster(tokens: torch.Tensor, rows: int, merge_size: int = 2, grid: tuple[int, int] | None = None) -> torch.Tensor:
    """Reorder (rows, h * w, width) tokens from merge-block order to raster order."""
    width = tokens.shape[-1]
    if grid is None:
        side = isqrt(tokens.numel() // (rows * width))
        grid = (side, side)
    height, wide = grid
    blocks = tokens.reshape(rows, height // merge_size, wide // merge_size, merge_size, merge_size, width)
    return blocks.permute(0, 1, 3, 2, 4, 5).reshape(rows, height * wide, width)


def stack_pixels(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.stack(samples)}


def pack_patches(samples: list[torch.Tensor], key: str = "pixel_values", grid_key: str = "grid_thw") -> dict[str, torch.Tensor]:
    side = isqrt(samples[0].shape[0])
    return {key: torch.cat(samples), grid_key: torch.tensor([[1, side, side]] * len(samples))}


@dataclass(frozen=True)
class ImagePreprocess:
    """Bench square crop, then ToTensor and optional Normalize, as a [C, H, W] image."""

    resolution: int
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    rgb: bool = False
    multiple: int = 1
    pipeline: transforms.Compose = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        step = self.grid_step()
        if self.resolution <= 0 or self.resolution % step:
            raise ValueError(f"resolution must be a positive multiple of {step}")
        steps = [square_crop(self.resolution), transforms.ToTensor()]
        if self.mean is not None:
            steps.append(transforms.Normalize(self.mean, self.std))
        object.__setattr__(self, "pipeline", transforms.Compose(steps))

    def grid_step(self) -> int:
        return self.multiple

    def pixels(self, image) -> torch.Tensor:
        return self.pipeline(image.convert("RGB") if self.rgb else image)

    def __call__(self, image) -> torch.Tensor:
        return self.pixels(image)


@dataclass(frozen=True)
class PatchPreprocess(ImagePreprocess):
    """Cut the image into patch rows grouped by merge block, Qwen2-VL style.

    Each row holds one patch as (C, T, P, P), with the still image repeated
    over T. ``raster`` undoes the merge-block grouping after the Tower.
    ``flatten=False`` keeps each row as [C * T, P, P].
    """

    patch_size: int = 14
    merge_size: int = 1
    temporal_patch_size: int = 1
    flatten: bool = True

    def grid_step(self) -> int:
        return self.patch_size * self.merge_size

    def __call__(self, image) -> torch.Tensor:
        pixels = self.pixels(image)
        channels, height, width = pixels.shape
        p, m, t = self.patch_size, self.merge_size, self.temporal_patch_size
        rows, cols = height // p, width // p
        frames = pixels.unsqueeze(0).expand(t, -1, -1, -1)
        patches = frames.reshape(1, t, channels, rows // m, m, p, cols // m, m, p)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        if self.flatten:
            return patches.reshape(rows * cols, channels * t * p * p)
        return patches.reshape(rows * cols, channels * t, p, p)
