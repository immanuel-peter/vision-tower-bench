"""Nemotron 3.5 vision components, using the existing C-RADIO implementation.

Only code is inherited from Nano Omni. Every tensor comes from the selected
3.5 checkpoint. Conditioner arithmetic stays in float32, as in the parents'
external image processor, before the BF16 vision forward.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vtb.adapters.nemotron_omni import (
    DOWNSAMPLE,
    PATCH_SIZE,
    PROJECTOR_HIDDEN,
    SOURCE_REPO as CODE_CONFIG_REPO,
    SOURCE_REVISION as CODE_CONFIG_REVISION,
    VIT_WIDTH,
    NemotronOmniAdapter,
    Preprocess,
    RMSNorm,
    SquaredReLU,
    attach_video_embedder,
    collate,
    pixel_shuffle,
)
from vtb.shards import load_prefixed

# AutoConfig's cross-repository auto_map otherwise follows the code repo's main.
RADIO_CODE_REVISION = "0057b339059c0b9e1b4ba996f975410ebbfdfcc8"
LLM_WIDTH = 4096
TOWER_PREFIX = "vision_model."
PROJECTOR_PREFIX = "mlp1."
FINAL_NORM_PREFIX = "vision_projector.vision_final_layernorm."
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str
    final_layernorm: bool

    @property
    def release_dir(self) -> Path:
        return ROOT / "hf" / self.model_id.split("/")[-1]

    @property
    def bundle(self) -> Path:
        override = os.environ.get(f"VTB_{self.name.upper()}_BUNDLE")
        return Path(override) if override else self.release_dir


SUPER = Checkpoint(
    "nemotron35_super",
    "immanuelpeter/C-RADIOv4-H-3.5-Super",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    "97e009758ae21747ee3752d2e43b005fb1e23a61",
    True,
)
BOLT = Checkpoint(
    "nemotron35_bolt",
    "immanuelpeter/C-RADIOv4-H-3.5-Bolt",
    "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16",
    "a6a1e5030819a130a1ea08aadb1fe42f4cc86777",
    False,
)
CHECKPOINTS = {item.name: item for item in (SUPER, BOLT)}


def source_index(checkpoint: Checkpoint) -> dict[str, str]:
    path = hf_hub_download(checkpoint.repo, "model.safetensors.index.json", revision=checkpoint.revision)
    return json.loads(Path(path).read_text())["weight_map"]


def source_weights(checkpoint: Checkpoint) -> dict[str, dict[str, torch.Tensor]]:
    index = source_index(checkpoint)
    parts = {}
    for part, prefix, count in (
        ("tower", TOWER_PREFIX, 391),
        ("projector", PROJECTOR_PREFIX, 3),
        ("final_norm", FINAL_NORM_PREFIX, 2 if checkpoint.final_layernorm else 0),
    ):
        names = {name for name in index if name.startswith(prefix)}
        if len(names) != count:
            raise ValueError(f"{checkpoint.name} {part}: expected {count} tensors, found {len(names)}")
        shards = sorted({index[name] for name in names})
        weights = load_prefixed(checkpoint.repo, shards, prefix, revision=checkpoint.revision) if shards else {}
        if set(weights) != {name.removeprefix(prefix) for name in names}:
            raise ValueError(f"{checkpoint.name} {part}: index/header inventory mismatch")
        parts[part] = weights
    return parts


def make_projector() -> nn.Sequential:
    return nn.Sequential(
        RMSNorm(VIT_WIDTH * 4, eps=1e-5),
        nn.Linear(VIT_WIDTH * 4, PROJECTOR_HIDDEN, bias=False),
        SquaredReLU(),
        nn.Linear(PROJECTOR_HIDDEN, LLM_WIDTH, bias=False),
    )


def build_parts(checkpoint: Checkpoint, weights: dict, dtype: torch.dtype):
    config = AutoConfig.from_pretrained(
        CODE_CONFIG_REPO, revision=CODE_CONFIG_REVISION, trust_remote_code=True,
    )
    radio_cls = get_class_from_dynamic_module(
        "hf_model.RADIOModel", "nvidia/C-RADIOv4-H", revision=RADIO_CODE_REVISION,
    )
    radio = radio_cls(config.vision_config)
    attach_video_embedder(radio)
    tower = dict(weights["tower"])
    summary = tower.pop("summary_idxs")
    if summary.dtype != torch.int64 or summary.tolist() != [0, 1]:
        raise ValueError("unexpected 3.5 summary token selection")
    radio.radio_model.summary_idxs = torch.empty_like(summary)
    tower["radio_model.summary_idxs"] = summary
    radio.load_state_dict(tower, strict=True)
    radio.to(dtype)
    # Parent composition detaches the conditioner. Its processor normalizes in
    # FP32, then the projector casts to BF16. Retaining FP32 buffers here and
    # passing FP32 pixels reproduces that boundary without double normalization.
    conditioner = radio.radio_model.input_conditioner.float()
    conditioner.load_state_dict({
        name: tower[f"radio_model.input_conditioner.{name}"]
        for name in ("norm_mean", "norm_std")
    }, strict=True)
    conditioner.dtype = dtype
    projector = make_projector()
    projector.load_state_dict(weights["projector"], strict=True)
    norm = nn.LayerNorm(VIT_WIDTH, eps=1e-6) if checkpoint.final_layernorm else nn.Identity()
    norm.load_state_dict(weights["final_norm"], strict=True)
    return radio.eval(), projector.to(dtype).eval(), norm.to(dtype).eval()


def load_parts(checkpoint: Checkpoint, dtype: torch.dtype):
    bundle = checkpoint.bundle
    if (bundle / "model.safetensors").exists():
        provenance = json.loads((bundle / "config.json").read_text())
        if provenance["source"] != checkpoint.repo or provenance["revision"] != checkpoint.revision:
            raise ValueError(f"bundle provenance does not match {checkpoint.name}")
        weights = {
            "tower": load_file(bundle / "model.safetensors"),
            "projector": load_file(bundle / "projector.safetensors"),
            "final_norm": load_file(bundle / "final_norm.safetensors") if checkpoint.final_layernorm else {},
        }
    else:
        weights = source_weights(checkpoint)
    return build_parts(checkpoint, weights, dtype)


class Nemotron35Adapter(NemotronOmniAdapter):
    checkpoint = SUPER
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution=448, dtype=torch.bfloat16, device="cuda", stages=None):
        if resolution <= 0 or resolution % (2 * PATCH_SIZE):
            raise ValueError("resolution must be a positive multiple of 32 for 2x2 pixel shuffle")
        self.selected_stages = tuple(self.stages if stages is None else stages)
        if not self.selected_stages or set(self.selected_stages) - set(self.stages):
            raise ValueError(f"unknown or empty stages: {self.selected_stages}")
        self.stages = self.selected_stages
        radio, projector, norm = load_parts(self.checkpoint, dtype)
        self.model, self.projector, self.final_norm = radio.to(device), projector.to(device), norm.to(device)
        self.model_id = self.checkpoint.model_id
        self.resolution, self.dtype, self.device = resolution, dtype, device
        self.num_layers = len(self.model.radio_model.model.blocks)

    @torch.inference_mode()
    def extract(self, inputs, image_ids):
        points = self.depth_points() if "tower" in self.selected_stages else []
        captured, handles = {}, []

        def capture(layer):
            def hook(_module, _args, output):
                captured[layer] = (output[0] if isinstance(output, tuple) else output).detach()
            return hook

        for layer in points:
            handles.append(self.model.radio_model.model.blocks[layer - 1].register_forward_hook(capture(layer)))
        pixels = inputs["pixel_values"].to(self.device, torch.float32)
        if pixels.shape[0] != len(image_ids) or tuple(pixels.shape[-2:]) != (self.resolution, self.resolution):
            for handle in handles:
                handle.remove()
            raise ValueError("image IDs or raster dimensions do not match the input")
        try:
            output = self.model(pixels)
        finally:
            for handle in handles:
                handle.remove()
        skip = self.model.radio_model.model.patch_generator.num_skip
        for layer in points:
            yield self._batch(captured[layer][:, skip:], image_ids, "tower", layer)
        if set(self.selected_stages) & {"merged", "projected"}:
            side = self.resolution // PATCH_SIZE
            tower = self.final_norm(output.features).reshape(len(image_ids), side, side, VIT_WIDTH)
            merged = pixel_shuffle(tower, DOWNSAMPLE).flatten(1, 2)
            if "merged" in self.selected_stages:
                yield self._batch(merged, image_ids, "merged", self.num_layers)
            if "projected" in self.selected_stages:
                yield self._batch(self.projector(merged), image_ids, "projected", self.num_layers)


class Nemotron35SuperAdapter(Nemotron35Adapter):
    checkpoint = SUPER


class Nemotron35BoltAdapter(Nemotron35Adapter):
    checkpoint = BOLT

    def __init__(self, *args, stages=("merged", "projected"), **kwargs):
        super().__init__(*args, stages=stages, **kwargs)
