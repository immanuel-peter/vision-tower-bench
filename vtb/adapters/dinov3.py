"""Pinned official DINOv3 ViT-L/16 control, separate from DINOv2."""

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

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
# Byte-for-byte copy of the pinned official checkpoint, as listed by the bundle manifest.
MODEL_FILE = "model.safetensors"
MODEL_BYTES = 1_212_559_808
MODEL_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
BUNDLE_ENV = "VTB_DINOV3_BUNDLE"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Path) -> dict:
    """Check a local copy of the pinned checkpoint before loading it.

    The bundle's source_manifest.json must name the pinned repository and
    revision, and every file it lists must match its recorded size and SHA256.
    The weights must also match the size and hash pinned here, so an edited
    manifest cannot vouch for different weights.
    """
    bundle = Path(bundle)
    manifest_path = bundle / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{bundle} has no source_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("source_repo"), manifest.get("source_revision")) != (SOURCE_REPO, SOURCE_REVISION):
        raise ValueError(f"{bundle} provenance does not match {SOURCE_REPO}@{SOURCE_REVISION}")
    files = manifest.get("files", {})
    if files.get(MODEL_FILE) != {"bytes": MODEL_BYTES, "sha256": MODEL_SHA256}:
        raise ValueError(f"{bundle} manifest does not list the pinned {MODEL_FILE}")
    for name in ("config.json", MODEL_FILE):
        if name not in files:
            raise ValueError(f"{bundle} manifest does not list {name}")
    for name, expected in files.items():
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(f"{bundle} is missing {name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{bundle}/{name}: size differs from the manifest")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"{bundle}/{name}: SHA256 differs from the manifest")
    return manifest


def bundle_path(bundle: Path | str | None = None) -> Path | None:
    """Explicit argument first, then $VTB_DINOV3_BUNDLE; None means the Hub."""
    if bundle is None:
        bundle = os.environ.get(BUNDLE_ENV) or None
    return None if bundle is None else Path(bundle)


def load_parent(dtype: torch.dtype, bundle: Path | str | None = None) -> DINOv3ViTModel:
    local = bundle_path(bundle)
    if local is None:
        return DINOv3ViTModel.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION, dtype=dtype)
    verify_bundle(local)
    return DINOv3ViTModel.from_pretrained(local, dtype=dtype, local_files_only=True)


class DINOv3Adapter:
    """Meta's DINOv3 ViT-L/16 as a second self-supervised control.

    The 512 px default gives a 32 by 32 patch grid, equal to DINOv2 at 448 px.
    Pass 448 for the equal-pixel check. The weights use the DINOv3 License, and
    publications using them must acknowledge Meta's DINO materials.

    ``bundle`` (or ``$VTB_DINOV3_BUNDLE``) loads a verified local copy of the
    same pinned checkpoint instead of the Hub; ``model_id`` stays the official
    repository either way.
    """

    model_id = SOURCE_REPO
    stages = ("tower",)
    collate = staticmethod(collate)

    def __init__(
        self, resolution: int = 512, dtype=torch.bfloat16, device: str = "mps", *,
        bundle: Path | str | None = None,
    ):
        check_resolution(resolution)
        self.model = load_parent(dtype, bundle).to(device).eval()
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
