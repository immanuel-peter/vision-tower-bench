"""Inkling 2 Small's vision Tower on timm's PE-Lang L Eva trunk, and its 2x2-merge Projector.

The checkpoint keeps the parent's tensor names (`backbone.*`, `proj_in`, `proj_out`). `to_timm` maps the
Tower onto `timm`'s `vit_pe_lang_large_patch14_448` at patch size 20, which runs the same 2D RoPE as
the parent: interleaved pairs, theta 10000, x then y, patch positions offset by one, class token at 0.
"""

import json
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

ARCHITECTURE = "vit_pe_lang_large_patch14_448"
PATCH_SIZE, MERGE_SIZE, POSITION_GRID = 20, 2, 32
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# The parent's image processor divides by float32-rounded CLIP std constants.
PROCESSOR_CLIP_STD = (0.26862954, 0.2613026, 0.2757771)
BLOCK_RENAMES = (
    ("self_attn.out_proj.", "attn.proj."),
    ("ln_1.", "norm1."),
    ("ln_2.", "norm2."),
    ("ls_1.gamma", "gamma_1"),
    ("ls_2.gamma", "gamma_2"),
    ("mlp.c_fc.", "mlp.fc1."),
    ("mlp.c_proj.", "mlp.fc2."),
)


def normalize(pixels: torch.Tensor) -> torch.Tensor:
    """[3, H, W] pixels in [0, 1] to the parent's input: CLIP-normalize, round to BF16, map to [-1, 1], round to BF16."""
    channel = (3, 1, 1)
    clip = ((pixels - torch.tensor(CLIP_MEAN).view(channel)) / torch.tensor(PROCESSOR_CLIP_STD).view(channel)).bfloat16().float()
    return (2.0 * (clip * torch.tensor(CLIP_STD).view(channel) + torch.tensor(CLIP_MEAN).view(channel)) - 1.0).bfloat16().float()


def to_timm(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the parent's `backbone.*` tensors onto timm Eva names, using each tensor exactly once."""
    remaining = dict(weights)

    def take(name: str) -> torch.Tensor:
        return remaining.pop("backbone." + name)

    width = weights["backbone.ln_pre.weight"].shape[0]
    patch = take("embeddings.patch_embedding.weight")
    side = int(round((patch.shape[1] // 3) ** 0.5))
    cls = take("embeddings.class_embedding")
    cls_position = take("embeddings.cls_position_embedding")
    positions = take("embeddings.position_embedding")
    out = {
        # The parent flattens each patch as (py, px, channel); a conv kernel is (channel, py, px).
        "patch_embed.proj.weight": patch.view(width, side, side, 3).permute(0, 3, 1, 2).contiguous(),
        "cls_token": cls.view(1, 1, width),
        "pos_embed": torch.cat([cls_position.view(1, width), positions]).unsqueeze(0),
        "norm_pre.weight": take("ln_pre.weight"),
        "norm_pre.bias": take("ln_pre.bias"),
    }
    layers = sorted({int(name.split(".")[2]) for name in remaining if name.startswith("backbone.layers.")})
    for index in layers:
        prefix = f"layers.{index}."
        for kind in ("weight", "bias"):
            out[f"blocks.{index}.attn.qkv.{kind}"] = torch.cat([take(f"{prefix}self_attn.{p}_proj.{kind}") for p in "qkv"])
        for name in [n.removeprefix("backbone." + prefix) for n in list(remaining) if n.startswith("backbone." + prefix)]:
            target = name
            for old, new in BLOCK_RENAMES:
                target = target.replace(old, new)
            if target == name:
                raise KeyError(f"no timm name for backbone.{prefix}{name}")
            out[f"blocks.{index}.{target}"] = take(prefix + name)
    if remaining:
        raise KeyError(f"unmapped Tower tensors: {sorted(remaining)[:5]}")
    return out


def merge(tokens: torch.Tensor, side: int) -> torch.Tensor:
    """[B, side * side, C] raster patch tokens to [B, (side / 2)^2, 4C]: each 2x2 block concatenated top-left, top-right, bottom-left, bottom-right."""
    rows, _, width = tokens.shape
    half = side // MERGE_SIZE
    blocks = tokens.reshape(rows, half, MERGE_SIZE, half, MERGE_SIZE, width).permute(0, 1, 3, 2, 4, 5)
    return blocks.reshape(rows, half * half, MERGE_SIZE * MERGE_SIZE * width)


class Projector(nn.Module):
    """`proj_out(gelu(proj_in(merged)))`, exact GELU, no biases."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.proj_in = nn.Linear(input_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, merged: torch.Tensor) -> torch.Tensor:
        return self.proj_out(F.gelu(self.proj_in(merged)))


def build_tower(weights: dict[str, torch.Tensor], patch_size: int = PATCH_SIZE) -> nn.Module:
    model = timm.create_model(ARCHITECTURE, pretrained=False, patch_size=patch_size, img_size=POSITION_GRID * patch_size)
    model.load_state_dict(to_timm(weights), strict=True)
    return model.eval()


def build_projector(weights: dict[str, torch.Tensor]) -> Projector:
    hidden, input_size = weights["proj_in.weight"].shape
    projector = Projector(input_size, hidden, weights["proj_out.weight"].shape[0])
    projector.load_state_dict(weights, strict=True)
    return projector.eval()


def load_vision(bundle: str | Path, dtype: torch.dtype = torch.float32) -> tuple[nn.Module, Projector]:
    """The Tower and Projector from a local release directory."""
    bundle = Path(bundle)
    config = json.loads((bundle / "config.json").read_text())
    tower = build_tower(load_file(bundle / "model.safetensors"), config["native_patch_size"])
    return tower.to(dtype), build_projector(load_file(bundle / "projector.safetensors")).to(dtype)
