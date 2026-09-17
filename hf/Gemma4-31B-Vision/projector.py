import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn


class Projector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.eps = config["norm_eps"]
        self.embedding_projection = nn.Linear(
            config["input_size"], config["output_size"], bias=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens * torch.rsqrt(tokens.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.embedding_projection(tokens)


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
