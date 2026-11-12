import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector


def load_projector(model: str | Path) -> Mistral3MultiModalProjector:
    """RMSNorm, 2x2 patch merger and two-layer GELU adapter, as in Mistral Large 4."""
    path = Path(model)
    if path.is_dir():
        config_path = path / "projector_config.json"
        weights_path = path / "projector.safetensors"
    else:
        config_path = Path(hf_hub_download(str(model), "projector_config.json"))
        weights_path = Path(hf_hub_download(str(model), "projector.safetensors"))
    projector = Mistral3MultiModalProjector(Mistral3Config(**json.loads(config_path.read_text())))
    projector.load_state_dict(load_file(weights_path), strict=True)
    return projector


def merged_and_projected(projector: Mistral3MultiModalProjector, tokens: torch.Tensor, image_sizes):
    """Return the patch-merger output (2048-d) and the Projector output (6144-d)."""
    merged = projector.patch_merger(projector.norm(tokens), image_sizes)
    return merged, projector.linear_2(projector.act(projector.linear_1(merged)))
