"""Standalone exact-GELU Projector for the pinned MoonViT vision bundle."""
import json
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config["input_size"], config["hidden_size"], bias=False),
            nn.GELU(),
            nn.Linear(config["hidden_size"], config["output_size"], bias=False),
        )
        self.post_norm = nn.RMSNorm(config["output_size"], eps=config["norm_eps"])

    def forward(self, tokens):
        return self.post_norm(self.proj(tokens))


def load_projector(root, *, dtype=torch.bfloat16, device="cpu"):
    root=Path(root)
    model=Projector(json.loads((root/"projector_config.json").read_text())).to(dtype=dtype)
    model.load_state_dict(load_file(root/"projector.safetensors"), strict=True)
    return model.to(device).eval()
