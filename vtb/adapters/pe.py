"""Perception Encoder Towers on timm's Eva trunk: Meta's PE-Lang L and Towers initialized from it."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, ImagePreprocess, default_device, depth_points, make_batch, stack_pixels
from vtb.feature_batch import FeatureBatch
from vtb.provenance import log_loaded

ARCHITECTURE = "vit_pe_lang_large_patch14_448"
POSITION_GRID = 32


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str
    patch_size: int

    @property
    def native_resolution(self) -> int:
        return POSITION_GRID * self.patch_size


PE_LANG_L = Checkpoint(
    "pe_lang_l",
    "timm/vit_pe_lang_large_patch14_448.fb",
    "timm/vit_pe_lang_large_patch14_448.fb",
    "9e668cb5b529f08c826fada8fc488bdcbf572970",
    14,
)
CHECKPOINTS = {item.name: item for item in (PE_LANG_L,)}

Preprocess = partial(ImagePreprocess, mean=HALF, std=HALF, rgb=True)
collate = stack_pixels


def timm_weights(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    return load_file(hf_hub_download(checkpoint.repo, "model.safetensors", revision=checkpoint.revision))


def load_tower(checkpoint: Checkpoint, dtype: torch.dtype) -> torch.nn.Module:
    model = timm.create_model(ARCHITECTURE, pretrained=False, patch_size=checkpoint.patch_size, img_size=checkpoint.native_resolution)
    model.load_state_dict(timm_weights(checkpoint), strict=True)
    return model.to(dtype).eval()


class PEAdapter:
    checkpoint: Checkpoint
    default_resolution = 448
    stages = ("tower",)
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        if self.resolution != self.checkpoint.native_resolution:
            raise ValueError(f"{self.checkpoint.name} runs on its native {POSITION_GRID}x{POSITION_GRID} grid at {self.checkpoint.native_resolution} px")
        self.device = default_device() if device is None else device
        self.dtype = dtype
        self.model_id = self.checkpoint.model_id
        log_loaded(self.model_id, f"{self.checkpoint.repo}@{self.checkpoint.revision}")
        self.model = load_tower(self.checkpoint, dtype).to(self.device)
        self.num_layers = len(self.model.blocks)

    def preprocess(self) -> ImagePreprocess:
        return Preprocess(self.resolution, multiple=self.checkpoint.patch_size)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    def tower(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Patch tokens after each sampled block, class token dropped, in the trunk's row-major order."""
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, self.resolution, self.resolution):
            raise ValueError(f"expected pixels [B, 3, {self.resolution}, {self.resolution}]")
        return self.model.forward_intermediates(
            pixel_values.to(self.device, self.dtype), indices=[layer - 1 for layer in self.depth_points()],
            output_fmt="NLC", intermediates_only=True,
        )

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        pixel_values = inputs["pixel_values"]
        if pixel_values.shape[0] != len(image_ids):
            raise ValueError("pixel batch and image_ids must have the same length")
        for layer, tokens in zip(self.depth_points(), self.tower(pixel_values), strict=True):
            yield make_batch(self, tokens, image_ids, "tower", layer)


class PELangLAdapter(PEAdapter):
    checkpoint = PE_LANG_L
