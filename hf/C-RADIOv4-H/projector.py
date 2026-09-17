import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * hidden_states).to(dtype)


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.nn.functional.relu(x), 2)


def load_projector(model: str | Path) -> nn.Sequential:
    path = Path(model)
    if path.is_dir():
        config_path = path / "projector_config.json"
        weights_path = path / "projector.safetensors"
    else:
        config_path = Path(hf_hub_download(str(model), "projector_config.json"))
        weights_path = Path(hf_hub_download(str(model), "projector.safetensors"))
    settings = json.loads(config_path.read_text())
    merged = settings["vit_hidden"] * 4
    projector = nn.Sequential(
        RMSNorm(merged, eps=1e-5),
        nn.Linear(merged, settings["projector_hidden"], bias=False),
        SquaredReLU(),
        nn.Linear(settings["projector_hidden"], settings["llm_hidden"], bias=False),
    )
    projector.load_state_dict(load_file(weights_path))
    return projector
