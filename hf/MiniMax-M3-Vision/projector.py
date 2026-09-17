import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn


class Projector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.spatial_merge_size = 2
        self.linear_1 = nn.Linear(config["input_size"], config["hidden_size"], bias=True)
        self.linear_2 = nn.Linear(config["hidden_size"], config["output_size"], bias=True)
        self.merge_linear_1 = nn.Linear(config["merged_hidden_size"], config["hidden_size"], bias=True)
        self.merge_linear_2 = nn.Linear(config["hidden_size"], config["output_size"], bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.linear_1(image_features))
        hidden = self.linear_2(hidden)
        hidden = hidden.reshape(hidden.shape[0] // (self.spatial_merge_size ** 2), -1)
        return self.merge_linear_2(torch.nn.functional.gelu(self.merge_linear_1(hidden)))


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
