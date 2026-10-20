"""Pinned official DINOv3 ViT-L/16 control, separate from DINOv2."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torchvision import transforms
from transformers import DINOv3ViTModel

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop

SOURCE_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SOURCE_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16
PREFIX_TOKENS = 5


def check_resolution(resolution: int) -> None:
    if resolution <= 0 or resolution % PATCH_SIZE:
        raise ValueError("DINOv3 resolution must be a positive multiple of 16")


@dataclass(frozen=True)
class Preprocess:
    """Bench crop and interpolation, official ImageNet channel normalization.

    The parent processor's default is a bilinear square resize. The bench uses
    a bicubic short-side resize and center crop for both DINO controls.
    """

    resolution: int
    pipeline: transforms.Compose = field(init=False, repr=False)

    def __post_init__(self) -> None:
        check_resolution(self.resolution)
        object.__setattr__(self, "pipeline", transforms.Compose([
            square_crop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]))

    def __call__(self, image) -> torch.Tensor:
        return self.pipeline(image.convert("RGB"))


def collate(samples: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(samples)


class DINOv3Adapter:
    """Meta's DINOv3 ViT-L/16 as a second self-supervised control.

    The 512 px default gives a 32 by 32 patch grid, equal to DINOv2 at 448 px.
    Pass 448 for the equal-pixel check. The weights use the DINOv3 License, and
    publications using them must acknowledge Meta's DINO materials.
    """

    model_id = SOURCE_REPO
    stages = ("tower",)
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 512, dtype=torch.bfloat16, device: str = "mps"):
        check_resolution(resolution)
        self.model = DINOv3ViTModel.from_pretrained(
            SOURCE_REPO, revision=SOURCE_REVISION, dtype=dtype,
        ).to(device).eval()
        config = self.model.config
        if (config.num_hidden_layers, config.hidden_size, config.patch_size,
                config.num_register_tokens) != (24, 1024, PATCH_SIZE, 4):
            raise ValueError("expected the pinned DINOv3 ViT-L/16 with four registers")
        self.resolution = resolution
        self.device = device
        self.num_layers = config.num_hidden_layers

    def preprocess(self) -> Preprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        if not 1 <= n <= self.num_layers:
            raise ValueError(f"depth count must be between 1 and {self.num_layers}")
        return [round(self.num_layers * (k + 1) / n) for k in range(n)]

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor, image_ids: list[str]) -> Iterator[FeatureBatch]:
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, self.resolution, self.resolution):
            raise ValueError(f"expected pixels [B, 3, {self.resolution}, {self.resolution}]")
        if pixel_values.shape[0] != len(image_ids):
            raise ValueError("pixel batch and image_ids must have the same length")
        out = self.model(
            pixel_values.to(self.device, self.model.dtype), output_hidden_states=True,
        )
        expected_tokens = (self.resolution // PATCH_SIZE) ** 2
        for layer in self.depth_points():
            # Match the DINOv2 bench convention: intermediate residual streams,
            # then the parent's learned final norm. Do not normalize twice.
            source = out.last_hidden_state if layer == self.num_layers else out.hidden_states[layer]
            patches = source[:, PREFIX_TOKENS:, :]
            if patches.shape[1:] != (expected_tokens, 1024):
                raise ValueError(f"unexpected DINOv3 patch layout {tuple(patches.shape)}")
            # Parent Conv2d flatten(2) emits rows first. Removing the prefix
            # preserves that raster order; no reshape or register averaging.
            yield FeatureBatch(
                tokens=patches.cpu(), image_ids=image_ids, model_id=self.model_id,
                stage="tower", layer_index=layer, num_layers=self.num_layers,
                resolution=self.resolution,
            )
