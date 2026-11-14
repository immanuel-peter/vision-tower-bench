"""Pinned official DINOv3 ViT-L/16 control, separate from DINOv2."""

from collections.abc import Iterator
from functools import partial

import torch
from transformers import DINOv3ViTModel

from vtb.adapters.base import Inputs
from vtb.adapters.common import IMAGENET_MEAN, IMAGENET_STD, ImagePreprocess, default_device, depth_points, make_batch, stack_pixels
from vtb.feature_batch import FeatureBatch
from vtb.provenance import log_loaded

SOURCE_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SOURCE_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
PATCH_SIZE = 16
PREFIX_TOKENS = 5

# Bench crop and interpolation with official ImageNet channel normalization. The
# parent processor's default is a bilinear square resize; the bench uses a
# bicubic short-side resize and center crop for both DINO controls.
Preprocess = partial(ImagePreprocess, mean=IMAGENET_MEAN, std=IMAGENET_STD, rgb=True, multiple=PATCH_SIZE)
collate = stack_pixels


class DINOv3Adapter:
    """Meta's DINOv3 ViT-L/16 as a second self-supervised control.

    The 512 px default gives a 32 by 32 patch grid, equal to DINOv2 at 448 px.
    Pass 448 for the equal-pixel check. The weights use the DINOv3 License, and
    publications using them must acknowledge Meta's DINO materials.
    """

    model_id = SOURCE_REPO
    default_resolution = 512
    stages = ("tower",)
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        Preprocess(self.resolution)
        log_loaded(self.model_id, f"{SOURCE_REPO}@{SOURCE_REVISION}")
        self.model = DINOv3ViTModel.from_pretrained(
            SOURCE_REPO, revision=SOURCE_REVISION, dtype=dtype,
        ).to(self.device).eval()
        config = self.model.config
        if (config.num_hidden_layers, config.hidden_size, config.patch_size,
                config.num_register_tokens) != (24, 1024, PATCH_SIZE, 4):
            raise ValueError("expected the pinned DINOv3 ViT-L/16 with four registers")
        self.num_layers = config.num_hidden_layers

    def preprocess(self) -> ImagePreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        pixel_values = inputs["pixel_values"]
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
            yield make_batch(self, patches, image_ids, "tower", layer)
