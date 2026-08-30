import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn


class Projector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config["input_size"], eps=config["norm_eps"])
        self.proj = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.GELU(),
            nn.Linear(config["hidden_size"], config["output_size"]),
        )

    def forward(self, groups: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.proj(self.pre_norm(group).flatten(1)) for group in groups]


def load_projector(model: str | Path) -> Projector:
    path = Path(model)
    if path.is_dir():
        config_path = path / "projector_config.json"
        weights_path = path / "projector.safetensors"
    else:
        config_path = Path(hf_hub_download(str(model), "projector_config.json"))
        weights_path = Path(hf_hub_download(str(model), "projector.safetensors"))
    projector = Projector(json.loads(config_path.read_text()))
    projector.load_state_dict(load_file(weights_path))
    return projector
