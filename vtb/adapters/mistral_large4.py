"""Mistral Large 4 Pixtral Tower and patch-merge Projector.

The parent ships only Mistral-native ``consolidated-*`` shards. Mistral applies
RoPE to interleaved (real, imaginary) channel pairs, while HF's Pixtral rotates
halves, so ``wq`` and ``wk`` are permuted per head while renaming. Every other
tensor is copied bit for bit.
"""

import json
import os
import re
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Mistral3Config, PixtralVisionConfig, PixtralVisionModel
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from vtb.adapters.base import Inputs
from vtb.adapters.common import ImagePreprocess, default_device, depth_points, make_batch, stack_pixels
from vtb.feature_batch import FeatureBatch
from vtb.models.pixtral_large4 import (  # noqa: F401  re-exported for tests and scripts
    ATTENTION_BACKENDS,
    PixtralLarge4VisionConfig,
    PixtralLarge4VisionModel,
    block_forward,
    encode,
    rotary_table,
    rotate,
)
from vtb.provenance import check_provenance, log_loaded
from vtb.shards import load_prefixed

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "immanuelpeter/PixtralViT-Large-4"
RELEASE_DIR = ROOT / "hf" / "PixtralViT-Large-4"
SOURCE_REPO = "mistralai/Mistral-Large-4-1T-A52B"
SOURCE_REVISION = "f12dabdc0c1e99e7f7924461a6f308859b05af4d"
SOURCE_INDEX = "consolidated.safetensors.index.json"
TOWER_PREFIX = "vision_encoder."
PROJECTOR_PREFIXES = ("pre_mm_projector_norm.", "patch_merger.", "vision_language_adapter.")
SOURCE_PREFIXES = (TOWER_PREFIX, *PROJECTOR_PREFIXES)
TOWER_TENSORS = 272
PROJECTOR_TENSORS = 4
PATCH_SIZE = 14
MERGE_SIZE = 2
ALIGN = PATCH_SIZE * MERGE_SIZE
NUM_HEADS = 16
# OpenAI CLIP statistics from the parent's PixtralImageProcessor.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

_LAYER = r"vision_encoder\.transformer\.layers\.(\d+)\."
_RULES = (
    (r"^vision_encoder\.patch_conv\.weight$", "vision_tower.patch_conv.weight"),
    (r"^vision_encoder\.ln_pre\.weight$", "vision_tower.ln_pre.weight"),
    (_LAYER + r"attention\.wq\.weight$", r"vision_tower.transformer.layers.\1.attention.q_proj.weight"),
    (_LAYER + r"attention\.wk\.weight$", r"vision_tower.transformer.layers.\1.attention.k_proj.weight"),
    (_LAYER + r"attention\.wv\.weight$", r"vision_tower.transformer.layers.\1.attention.v_proj.weight"),
    (_LAYER + r"attention\.wo\.weight$", r"vision_tower.transformer.layers.\1.attention.o_proj.weight"),
    (_LAYER + r"feed_forward\.w1\.weight$", r"vision_tower.transformer.layers.\1.feed_forward.gate_proj.weight"),
    (_LAYER + r"feed_forward\.w2\.weight$", r"vision_tower.transformer.layers.\1.feed_forward.down_proj.weight"),
    (_LAYER + r"feed_forward\.w3\.weight$", r"vision_tower.transformer.layers.\1.feed_forward.up_proj.weight"),
    (_LAYER + r"attention_norm\.weight$", r"vision_tower.transformer.layers.\1.attention_norm.weight"),
    (_LAYER + r"ffn_norm\.weight$", r"vision_tower.transformer.layers.\1.ffn_norm.weight"),
    (r"^pre_mm_projector_norm\.weight$", "multi_modal_projector.norm.weight"),
    (r"^patch_merger\.merging_layer\.weight$", "multi_modal_projector.patch_merger.merging_layer.weight"),
    (r"^vision_language_adapter\.w_in\.weight$", "multi_modal_projector.linear_1.weight"),
    (r"^vision_language_adapter\.w_out\.weight$", "multi_modal_projector.linear_2.weight"),
)
ROTARY_TARGETS = (".attention.q_proj.weight", ".attention.k_proj.weight")


def remap_key(name: str) -> str:
    """Return the HF name for one Mistral-native vision tensor, or raise."""
    for pattern, target in _RULES:
        renamed, count = re.subn(pattern, target, name)
        if count:
            return renamed
    raise KeyError(f"no HF name for {name}")


def is_rotary(hf_name: str) -> bool:
    return hf_name.endswith(ROTARY_TARGETS)


def interleaved_to_halves(weight: torch.Tensor, heads: int = NUM_HEADS) -> torch.Tensor:
    """Reorder each head's output rows from (re, im) pairs to [re..., im...].

    Mistral rotates channel pairs (2i, 2i + 1) as one complex number. HF rotates
    channel i with channel i + head_dim / 2, so pair member 0 goes to the first
    half and member 1 to the second.
    """
    rows, cols = weight.shape
    half = rows // heads // 2
    return weight.view(heads, half, 2, cols).transpose(1, 2).reshape(rows, cols)


def halves_to_interleaved(weight: torch.Tensor, heads: int = NUM_HEADS) -> torch.Tensor:
    rows, cols = weight.shape
    half = rows // heads // 2
    return weight.view(heads, 2, half, cols).transpose(1, 2).reshape(rows, cols)


def convert(native: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split native tensors into HF Tower and Projector state dicts."""
    tower, projector = {}, {}
    for name, tensor in native.items():
        hf_name = remap_key(name)
        if is_rotary(hf_name):
            tensor = interleaved_to_halves(tensor)
        if hf_name.startswith("vision_tower."):
            tower[hf_name.removeprefix("vision_tower.")] = tensor
        else:
            projector[hf_name.removeprefix("multi_modal_projector.")] = tensor
    if len(tower) != TOWER_TENSORS or len(projector) != PROJECTOR_TENSORS:
        raise ValueError(f"expected {TOWER_TENSORS} + {PROJECTOR_TENSORS} tensors, got {len(tower)} + {len(projector)}")
    return tower, projector


def source_json(filename: str) -> dict:
    return json.loads(Path(hf_hub_download(SOURCE_REPO, filename, revision=SOURCE_REVISION)).read_text())


def source_index() -> dict[str, str]:
    """Vision tensor name to shard, discovered from the pinned index by prefix."""
    weight_map = source_json(SOURCE_INDEX)["weight_map"]
    return {name: shard for name, shard in weight_map.items() if name.startswith(SOURCE_PREFIXES)}


def source_weights() -> dict[str, torch.Tensor]:
    """Fetch only the vision tensors, by byte range, under their native names."""
    index = source_index()
    native = {}
    for prefix in SOURCE_PREFIXES:
        shards = sorted({shard for name, shard in index.items() if name.startswith(prefix)})
        for name, tensor in load_prefixed(SOURCE_REPO, shards, prefix, revision=SOURCE_REVISION).items():
            native[prefix + name] = tensor
    if set(native) != set(index):
        raise ValueError(f"vision tensors differ from the pinned index: {sorted(set(index) ^ set(native))[:5]}")
    if any(tensor.dtype != torch.bfloat16 for tensor in native.values()):
        raise ValueError("every Mistral Large 4 vision tensor should be BF16")
    return native


def vision_config(source: dict | None = None) -> PixtralLarge4VisionConfig:
    source = source or source_json("config.json")
    settings = dict(source["vision_config"])
    settings.pop("model_type", None)
    return PixtralLarge4VisionConfig(**settings)


def projector_config(source: dict | None = None) -> Mistral3Config:
    source = source or source_json("config.json")
    text = source["text_config"]
    vision = PixtralVisionConfig(**{k: v for k, v in source["vision_config"].items() if k != "model_type"})
    return Mistral3Config(
        vision_config=vision.to_dict(),
        text_config={"model_type": "mistral", "hidden_size": text["hidden_size"], "rms_norm_eps": text["rms_norm_eps"]},
        spatial_merge_size=source["spatial_merge_size"],
        projector_hidden_act=source["projector_hidden_act"],
        multimodal_projector_bias=source["multimodal_projector_bias"],
        vision_feature_layer=source["vision_feature_layer"],
    )


def build_parts(
    tower_state: dict[str, torch.Tensor],
    projector_state: dict[str, torch.Tensor],
    tower_config: PixtralLarge4VisionConfig,
    mm_config: Mistral3Config,
    dtype: torch.dtype,
) -> tuple[PixtralLarge4VisionModel, Mistral3MultiModalProjector]:
    tower = PixtralLarge4VisionModel._from_config(tower_config, dtype=dtype)
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in tower_state.items()}, strict=True)
    projector = Mistral3MultiModalProjector(mm_config).to(dtype)
    projector.load_state_dict({name: tensor.to(dtype) for name, tensor in projector_state.items()}, strict=True)
    # RoPE frequencies stay float32 under BF16 weights, as HF from_pretrained keeps them.
    rotary = tower.patch_positional_embedding
    inv_freq, _ = rotary.compute_axial_rope_parameters(tower_config)
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()
    return tower.eval(), projector.eval()


def load_source_parts(dtype: torch.dtype = torch.bfloat16) -> tuple[PixtralVisionModel, Mistral3MultiModalProjector]:
    source = source_json("config.json")
    tower_state, projector_state = convert(source_weights())
    return build_parts(tower_state, projector_state, vision_config(source), projector_config(source), dtype)


def bundle_dir() -> Path:
    override = os.environ.get("VTB_MISTRAL_LARGE4_BUNDLE")
    return Path(override) if override else RELEASE_DIR


def load_bundle_parts(bundle: Path, dtype: torch.dtype) -> tuple[PixtralVisionModel, Mistral3MultiModalProjector]:
    check_provenance(bundle, SOURCE_REPO, SOURCE_REVISION, required=True)
    tower_config = PixtralLarge4VisionConfig.from_pretrained(bundle, local_files_only=True)
    mm_config = Mistral3Config(**json.loads((bundle / "projector_config.json").read_text()))
    return build_parts(
        load_file(bundle / "model.safetensors"),
        load_file(bundle / "projector.safetensors"),
        tower_config,
        mm_config,
        dtype,
    )


def load_bundle_projector(bundle: Path, dtype: torch.dtype) -> Mistral3MultiModalProjector:
    projector = Mistral3MultiModalProjector(Mistral3Config(**json.loads((bundle / "projector_config.json").read_text())))
    projector.load_state_dict(load_file(bundle / "projector.safetensors"), strict=True)
    return projector.to(dtype).eval()


def load_parts(dtype: torch.dtype) -> tuple[PixtralVisionModel, Mistral3MultiModalProjector]:
    bundle = bundle_dir()
    if (bundle / "model.safetensors").is_file():
        log_loaded(MODEL_ID, str(bundle))
        return load_bundle_parts(bundle, dtype)
    log_loaded(MODEL_ID, f"parent checkpoint {SOURCE_REPO}@{SOURCE_REVISION}")
    return load_source_parts(dtype)


# Square crop with bicubic resize, then CLIP normalization. No padding is needed.
Preprocess = partial(ImagePreprocess, mean=IMAGE_MEAN, std=IMAGE_STD, rgb=True, multiple=ALIGN)


def collate(samples: list[torch.Tensor]) -> Inputs:
    if not samples or any(sample.shape != samples[0].shape for sample in samples):
        raise ValueError("Mistral Large 4 batches require equal square images")
    return stack_pixels(samples)


def tower_forward(
    tower: PixtralVisionModel,
    pixel_values: torch.Tensor,
    capture: tuple[int, ...] = (),
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """The release's own forward (vtb/models/pixtral_large4.py), in the parent's numeric order."""
    return encode(tower, pixel_values, capture)


def projector_stages(
    projector: Mistral3MultiModalProjector, tokens: torch.Tensor, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (merged, projected) for a (batch, side * side, width) raster Tower output."""
    rows, _, width = tokens.shape
    pixels = side * PATCH_SIZE
    flat = projector.norm(tokens.reshape(-1, width))
    merged = projector.patch_merger(flat, [(pixels, pixels)] * rows)
    projected = projector.linear_2(projector.act(projector.linear_1(merged)))
    return merged.view(rows, -1, merged.shape[-1]), projected.view(rows, -1, projected.shape[-1])


class MistralLarge4Adapter:
    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        Preprocess(self.resolution)  # Reject unsupported grids before loading weights.
        tower, projector = load_parts(dtype)
        self.model = tower.to(self.device)
        self.projector = projector.to(self.device)
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> ImagePreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        if len(pixels) != len(image_ids):
            raise ValueError("pixel/image ID count differs")
        points = self.depth_points()
        final, captured = tower_forward(self.model, pixels, tuple(points))
        for layer in points:
            yield make_batch(self, captured[layer], image_ids, "tower", layer)
        merged, projected = projector_stages(self.projector, final, pixels.shape[-1] // PATCH_SIZE)
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        yield make_batch(self, projected, image_ids, "projected", self.num_layers)
