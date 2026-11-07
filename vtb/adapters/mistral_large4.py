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
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Mistral3Config, PixtralVisionConfig, PixtralVisionModel
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
from transformers.models.pixtral.modeling_pixtral import rotate_half

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
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


def vision_config(source: dict | None = None) -> PixtralVisionConfig:
    source = source or source_json("config.json")
    settings = dict(source["vision_config"])
    settings.pop("model_type", None)
    return PixtralVisionConfig(**settings)


def projector_config(source: dict | None = None) -> Mistral3Config:
    source = source or source_json("config.json")
    text = source["text_config"]
    return Mistral3Config(
        vision_config=vision_config(source).to_dict(),
        text_config={"model_type": "mistral", "hidden_size": text["hidden_size"], "rms_norm_eps": text["rms_norm_eps"]},
        spatial_merge_size=source["spatial_merge_size"],
        projector_hidden_act=source["projector_hidden_act"],
        multimodal_projector_bias=source["multimodal_projector_bias"],
        vision_feature_layer=source["vision_feature_layer"],
    )


def build_parts(
    tower_state: dict[str, torch.Tensor],
    projector_state: dict[str, torch.Tensor],
    tower_config: PixtralVisionConfig,
    mm_config: Mistral3Config,
    dtype: torch.dtype,
) -> tuple[PixtralVisionModel, Mistral3MultiModalProjector]:
    tower = PixtralVisionModel._from_config(tower_config, dtype=dtype)
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
    manifest = json.loads((bundle / "export_manifest.json").read_text())
    if (manifest["source_repo"], manifest["source_revision"]) != (SOURCE_REPO, SOURCE_REVISION):
        raise ValueError("bundle provenance does not match the pinned Mistral Large 4 revision")
    tower_config = PixtralVisionConfig.from_pretrained(bundle, local_files_only=True)
    mm_config = Mistral3Config(**json.loads((bundle / "projector_config.json").read_text()))
    return build_parts(
        load_file(bundle / "model.safetensors"),
        load_file(bundle / "projector.safetensors"),
        tower_config,
        mm_config,
        dtype,
    )


def load_parts(dtype: torch.dtype) -> tuple[PixtralVisionModel, Mistral3MultiModalProjector]:
    bundle = bundle_dir()
    if (bundle / "model.safetensors").is_file():
        return load_bundle_parts(bundle, dtype)
    return load_source_parts(dtype)


@dataclass(frozen=True)
class Preprocess:
    """Square crop with bicubic resize, then CLIP normalization. No padding is needed."""

    resolution: int
    pipeline: transforms.Compose = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.resolution <= 0 or self.resolution % ALIGN:
            raise ValueError(f"Mistral Large 4 resolution must be a positive multiple of {ALIGN}")
        object.__setattr__(self, "pipeline", transforms.Compose([
            square_crop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]))

    def __call__(self, image) -> torch.Tensor:
        return self.pipeline(image.convert("RGB"))


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    if not samples or any(sample.shape != samples[0].shape for sample in samples):
        raise ValueError("Mistral Large 4 batches require equal square images")
    return {"pixel_values": torch.stack(samples)}


# Mistral's attention calls xformers memory_efficient_attention. PyTorch's
# memory-efficient SDPA kernel is adapted from it; MATH is the CPU fallback.
ATTENTION_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def rotary_table(rows: int, cols: int, head_dim: int, theta: float) -> torch.Tensor:
    """Complex rotations per (row, col) position, built as mistral_inference.rope builds them.

    Float32 on the CPU, then torch.polar, exactly as precompute_freqs_cis_2d
    does. Entry j rotates the pair (j, j + head_dim / 2) of the HF layout,
    which is the parent's pair (2j, 2j + 1) after the export permutation.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs_h = torch.outer(torch.arange(rows), freqs[::2]).float()
    freqs_w = torch.outer(torch.arange(cols), freqs[1::2]).float()
    grid = torch.cat([freqs_h[:, None, :].repeat(1, cols, 1), freqs_w[None, :, :].repeat(rows, 1, 1)], dim=-1)
    return torch.polar(torch.ones_like(grid), grid).reshape(rows * cols, head_dim // 2)


def rotate(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in float32, then a cast back, as mistral_inference does.

    HF Pixtral instead casts cos/sin to the activation dtype and rotates in
    BF16, which changes about a quarter of the rotated BF16 values.
    """
    half = x.shape[-1] // 2
    x32 = x.float()
    turned = torch.complex(x32[..., :half], x32[..., half:]) * table
    return torch.cat((turned.real, turned.imag), dim=-1).to(x.dtype)


def block_forward(layer, hidden: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """One PixtralAttentionLayer with the parent's numeric order for RoPE and attention."""
    attention = layer.attention
    batch, tokens, _ = hidden.shape
    normed = layer.attention_norm(hidden)

    def heads(projection):
        return projection(normed).view(batch, tokens, attention.num_heads, attention.head_dim).transpose(1, 2)

    query = rotate(heads(attention.q_proj), table)
    key = rotate(heads(attention.k_proj), table)
    value = heads(attention.v_proj)
    with sdpa_kernel(ATTENTION_BACKENDS):
        mixed = F.scaled_dot_product_attention(query, key, value, scale=attention.scaling)
    hidden = hidden + attention.o_proj(mixed.transpose(1, 2).reshape(batch, tokens, -1))
    return hidden + layer.feed_forward(layer.ffn_norm(hidden))


def tower_forward(
    tower: PixtralVisionModel,
    pixel_values: torch.Tensor,
    capture: tuple[int, ...] = (),
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Run equal-size images as a batch instead of one block-diagonal sequence.

    The parent packs every image into one sequence with a block-diagonal mask,
    so images never attend to each other. With equal sizes that is the same as
    an unmasked batch; tests check the two paths against each other. RoPE and
    attention follow the parent's numeric order (see block_forward).
    """
    embeds = tower.patch_conv(pixel_values.to(tower.patch_conv.weight.dtype))
    rows, cols = embeds.shape[-2:]
    hidden = tower.ln_pre(embeds.flatten(2).transpose(1, 2))
    config = tower.config
    table = rotary_table(rows, cols, config.head_dim, config.rope_parameters["rope_theta"]).to(hidden.device)
    captured = {}
    for depth, layer in enumerate(tower.transformer.layers, start=1):
        hidden = block_forward(layer, hidden, table)
        if depth in capture:
            captured[depth] = hidden
    return hidden, captured


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
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        Preprocess(resolution)  # Reject unsupported grids before loading weights.
        tower, projector = load_parts(dtype)
        self.model = tower.to(device)
        self.projector = projector.to(device)
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> Preprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return [round(self.num_layers * (k + 1) / n) for k in range(n)]

    @torch.inference_mode()
    def extract(self, inputs: dict[str, torch.Tensor], image_ids: list[str]) -> Iterator[FeatureBatch]:
        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        if len(pixels) != len(image_ids):
            raise ValueError("pixel/image ID count differs")
        points = self.depth_points()
        final, captured = tower_forward(self.model, pixels, tuple(points))
        for layer in points:
            yield self._batch(captured[layer], image_ids, "tower", layer)
        merged, projected = projector_stages(self.projector, final, pixels.shape[-1] // PATCH_SIZE)
        yield self._batch(merged, image_ids, "merged", self.num_layers)
        yield self._batch(projected, image_ids, "projected", self.num_layers)

    def _batch(self, tokens: torch.Tensor, image_ids: list[str], stage: str, layer: int) -> FeatureBatch:
        return FeatureBatch(
            tokens=tokens.cpu(),
            image_ids=image_ids,
            model_id=self.model_id,
            stage=stage,
            layer_index=layer,
            num_layers=self.num_layers,
            resolution=self.resolution,
        )

