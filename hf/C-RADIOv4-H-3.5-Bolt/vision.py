"""Load the immanuelpeter/C-RADIOv4-H-3.5-Bolt vision bundle.

No language model weights are requested.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
        return (self.weight.float() * value).to(x.dtype)


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


def pixel_shuffle(x):
    n, h, w, c = x.shape
    x = x.reshape(n, h, w // 2, c * 2).permute(0, 2, 1, 3).contiguous()
    return x.reshape(n, w // 2, h // 2, c * 4).permute(0, 2, 1, 3).contiguous()


class Vision(nn.Module):
    def __init__(self, bundle, dtype=torch.bfloat16):
        super().__init__()
        bundle = Path(bundle)
        self.settings = json.loads((bundle / "config.json").read_text())
        config = AutoConfig.from_pretrained(
            self.settings["code_config_source"], revision=self.settings["code_config_revision"], trust_remote_code=True,
        )
        radio_cls = get_class_from_dynamic_module(
            "hf_model.RADIOModel", "nvidia/C-RADIOv4-H", revision=self.settings["radio_code_revision"],
        )
        self.tower = radio_cls(config.vision_config)
        generator = self.tower.radio_model.model.patch_generator
        generator.video_embedder = nn.Linear(2 * 3 * 16 * 16, 1280, bias=False)
        weights = load_file(bundle / "model.safetensors")
        self.tower.radio_model.summary_idxs = weights.pop("summary_idxs")
        weights["radio_model.summary_idxs"] = self.tower.radio_model.summary_idxs
        self.tower.load_state_dict(weights, strict=True)
        self.tower.to(dtype)
        conditioner = self.tower.radio_model.input_conditioner
        conditioner.float()
        conditioner.load_state_dict({
            name: weights[f"radio_model.input_conditioner.{name}"]
            for name in ("norm_mean", "norm_std")
        }, strict=True)
        conditioner.dtype = dtype
        self.final_norm = nn.LayerNorm(1280, eps=1e-6) if self.settings["vision_final_layernorm"] else nn.Identity()
        if self.settings["vision_final_layernorm"]:
            self.final_norm.load_state_dict(load_file(bundle / "final_norm.safetensors"), strict=True)
        self.final_norm.to(dtype)
        self.projector = nn.Sequential(
            RMSNorm(5120), nn.Linear(5120, 20480, bias=False),
            SquaredReLU(), nn.Linear(20480, 4096, bias=False),
        )
        self.projector.load_state_dict(load_file(bundle / "projector.safetensors"), strict=True)
        self.projector.to(dtype)
        self.eval()

    @torch.inference_mode()
    def forward(self, pixel_values):
        """Accept RGB float32 pixels in [0, 1] on a grid divisible by 32."""
        h, w = pixel_values.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("pixel dimensions must be divisible by 32")
        tower = self.tower(pixel_values.float()).features
        merged = pixel_shuffle(self.final_norm(tower).reshape(len(tower), h // 16, w // 16, 1280)).flatten(1, 2)
        return {"tower": tower, "merged": merged, "projected": self.projector(merged)}


def load_vision(bundle, dtype=torch.bfloat16):
    return Vision(bundle, dtype=dtype)
