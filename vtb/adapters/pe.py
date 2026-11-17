"""Perception Encoder Towers on timm's Eva trunk: Meta's PE-Lang L and Towers initialized from it."""

import json
import os
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, ImagePreprocess, default_device, depth_points, make_batch, stack_pixels
from vtb.bundle import ROOT, write_json
from vtb.feature_batch import FeatureBatch
from vtb.models import inkling2
from vtb.provenance import log_loaded, open_release
from vtb.shards import load_prefixed

ARCHITECTURE = "vit_pe_lang_large_patch14_448"
DEPTH, WIDTH = 23, 1024
POSITION_GRID = 32
VISUAL_PREFIX = "model.visual."
TOWER_TENSORS, PROJECTOR_TENSORS = 420, 2
MODELING_SOURCE = ROOT / "vtb/models/inkling2.py"
# vLLM's open Inkling 2 PR at the head whose vision path the parity reference vendors.
VLLM_REFERENCE = ("vllm-project/vllm", 67325, "d664cd1e96052857aea2aa844eeba571c24758b4")


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str
    patch_size: int
    merge_size: int = 1

    @property
    def native_resolution(self) -> int:
        return POSITION_GRID * self.patch_size

    @property
    def release_dir(self) -> Path:
        return ROOT / "hf" / self.model_id.split("/")[-1]

    def open(self) -> Path:
        override = os.environ.get(f"VTB_{self.name.upper()}_BUNDLE")
        return open_release(self.model_id, self.repo, self.revision, bundle=Path(override) if override else None)


PE_LANG_L = Checkpoint(
    "pe_lang_l",
    "timm/vit_pe_lang_large_patch14_448.fb",
    "timm/vit_pe_lang_large_patch14_448.fb",
    "9e668cb5b529f08c826fada8fc488bdcbf572970",
    14,
)
INKLING2_SMALL = Checkpoint(
    "inkling2_small",
    "immanuelpeter/Inkling-2-Small-Vision",
    "thinkingmachines/Inkling-2-Small",
    "b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b",
    inkling2.PATCH_SIZE,
    inkling2.MERGE_SIZE,
)
CHECKPOINTS = {item.name: item for item in (PE_LANG_L, INKLING2_SMALL)}

Preprocess = partial(ImagePreprocess, mean=HALF, std=HALF, rgb=True)
collate = stack_pixels


@dataclass(frozen=True)
class InklingPreprocess(ImagePreprocess):
    """The bench crop, then the parent's two BF16 roundings on the way to [-1, 1]."""

    def __call__(self, image) -> torch.Tensor:
        return inkling2.normalize(self.pixels(image))


def timm_weights(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    return load_file(hf_hub_download(checkpoint.repo, "model.safetensors", revision=checkpoint.revision))


def source_json(checkpoint: Checkpoint, filename: str) -> dict:
    return json.loads(Path(hf_hub_download(checkpoint.repo, filename, revision=checkpoint.revision)).read_text())


def source_weights(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    """Range-read the `model.visual.*` tensors from the shards the pinned index names, with the prefix stripped."""
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    selected = {name for name in index if name.startswith(VISUAL_PREFIX)}
    shards = sorted({index[name] for name in selected})
    weights = load_prefixed(checkpoint.repo, shards, VISUAL_PREFIX, revision=checkpoint.revision)
    if {VISUAL_PREFIX + name for name in weights} != selected or len(weights) != TOWER_TENSORS + PROJECTOR_TENSORS:
        raise ValueError(f"incomplete {checkpoint.name} visual state: {len(weights)} of {len(selected)} tensors")
    return weights


def split(weights: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tower = {name: tensor for name, tensor in weights.items() if name.startswith("backbone.")}
    projector = {name: tensor for name, tensor in weights.items() if name in ("proj_in.weight", "proj_out.weight")}
    if len(tower) != TOWER_TENSORS or len(projector) != PROJECTOR_TENSORS or len(tower) + len(projector) != len(weights):
        raise ValueError(f"expected {TOWER_TENSORS} Tower and {PROJECTOR_TENSORS} Projector tensors, got {len(tower)} and {len(projector)} of {len(weights)}")
    return tower, projector


def remap_checks(weights: dict[str, torch.Tensor]) -> dict:
    """What parity cannot see: every tensor mapped once, the reshaped shapes, and weights that are not PE-Lang L's."""
    tower, projector = split(weights)
    mapped = inkling2.to_timm(tower)
    shapes = {
        "patch_embed.proj.weight": (list(mapped["patch_embed.proj.weight"].shape), [WIDTH, 3, 20, 20]),
        "pos_embed": (list(mapped["pos_embed"].shape), [1, POSITION_GRID**2 + 1, WIDTH]),
        "proj_in.weight": (list(projector["proj_in.weight"].shape), [4096, 4096]),
        "proj_out.weight": (list(projector["proj_out.weight"].shape), [4608, 4096]),
    }
    pe_lang = timm_weights(PE_LANG_L)
    compared = [name for name, tensor in mapped.items() if name in pe_lang and pe_lang[name].shape == tensor.shape]
    equal = [name for name in compared if torch.equal(pe_lang[name].to(mapped[name].dtype), mapped[name])]
    report = {
        "source_tensors": len(weights), "mapped_to_timm": len(tower), "timm_tensors": len(mapped), "projector_tensors": len(projector),
        "timm_names_match": set(mapped) == set(pe_lang),
        "shapes": {name: {"actual": actual, "expected": expected} for name, (actual, expected) in shapes.items()},
        "pe_lang_compared": len(compared), "pe_lang_equal": equal,
    }
    report["passed"] = (len(weights) == TOWER_TENSORS + PROJECTOR_TENSORS and report["timm_names_match"] and not equal
                        and set(mapped) - set(compared) == {"patch_embed.proj.weight"}
                        and all(actual == expected for actual, expected in shapes.values()))
    return report


def export(out: Path, checkpoint: Checkpoint) -> None:
    if checkpoint is not INKLING2_SMALL:
        raise SystemExit(f"{checkpoint.name} is loaded from its upstream release and has no export")
    weights = source_weights(checkpoint)
    checks = remap_checks(weights)
    if not checks["passed"]:
        raise ValueError(f"remap checks failed: {json.dumps(checks)}")
    tower, projector = split(weights)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() != checkpoint.release_dir.resolve():
        shutil.copytree(checkpoint.release_dir, out, ignore=shutil.ignore_patterns("*.safetensors"), dirs_exist_ok=True)
    save_file(tower, out / "model.safetensors", metadata={"format": "pt"})
    save_file(projector, out / "projector.safetensors", metadata={"format": "pt"})
    shutil.copyfile(MODELING_SOURCE, out / "inkling2_vision.py")
    config = source_json(checkpoint, "config.json")
    write_json(out / "config.json", {
        **config["vision_config"], "text_hidden_size": config["text_config"]["hidden_size"], "image_size": checkpoint.native_resolution,
        "timm_architecture": ARCHITECTURE, "torch_dtype": "bfloat16",
    })
    repo, number, head = VLLM_REFERENCE
    write_json(out / "export_manifest.json", {
        "source_repo": checkpoint.repo, "source_revision": checkpoint.revision,
        "tensors": len(weights), "tower_tensors": len(tower), "projector_tensors": len(projector),
        "bytes": sum(tensor.nbytes for tensor in weights.values()),
        "vision_reference": f"https://github.com/{repo}/pull/{number} at {head}",
    })


def load_tower(checkpoint: Checkpoint, weights: Callable[[], dict[str, torch.Tensor]], dtype: torch.dtype) -> torch.nn.Module:
    model = timm.create_model(ARCHITECTURE, pretrained=False, patch_size=checkpoint.patch_size, img_size=checkpoint.native_resolution)
    model.load_state_dict(weights(), strict=True)
    return model.to(dtype).eval()


class PEAdapter:
    checkpoint: Checkpoint
    default_resolution = 448
    stages = ("tower",)
    collate = staticmethod(collate)
    Preprocess = staticmethod(Preprocess)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        if self.resolution != self.checkpoint.native_resolution:
            raise ValueError(f"{self.checkpoint.name} runs on its native {POSITION_GRID}x{POSITION_GRID} grid at {self.checkpoint.native_resolution} px")
        self.device = default_device() if device is None else device
        self.dtype = dtype
        self.model_id = self.checkpoint.model_id
        self.model = load_tower(self.checkpoint, self.trunk_weights, dtype).to(self.device)
        self.num_layers = len(self.model.blocks)

    def trunk_weights(self) -> dict[str, torch.Tensor]:
        log_loaded(self.model_id, f"{self.checkpoint.repo}@{self.checkpoint.revision}")
        return timm_weights(self.checkpoint)

    def preprocess(self) -> ImagePreprocess:
        return self.Preprocess(self.resolution, multiple=self.checkpoint.patch_size * self.checkpoint.merge_size)

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


class Inkling2SmallAdapter(PEAdapter):
    checkpoint = INKLING2_SMALL
    default_resolution = INKLING2_SMALL.native_resolution
    stages = ("tower", "merged", "projected")
    Preprocess = staticmethod(partial(InklingPreprocess, rgb=True))

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        super().__init__(resolution, dtype, device)
        self.projector = inkling2.build_projector(load_file(self.bundle / "projector.safetensors")).to(self.device, dtype)

    def trunk_weights(self) -> dict[str, torch.Tensor]:
        self.bundle = self.checkpoint.open()
        return inkling2.to_timm(load_file(self.bundle / "model.safetensors"))

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        pixel_values = inputs["pixel_values"]
        if pixel_values.shape[0] != len(image_ids):
            raise ValueError("pixel batch and image_ids must have the same length")
        sampled = self.tower(pixel_values)
        for layer, tokens in zip(self.depth_points(), sampled, strict=True):
            yield make_batch(self, tokens, image_ids, "tower", layer)
        merged = inkling2.merge(sampled[-1], self.resolution // self.checkpoint.patch_size)
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        yield make_batch(self, self.projector(merged), image_ids, "projected", self.num_layers)
