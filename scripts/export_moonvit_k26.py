"""Build a self-contained Kimi K2.6 MoonViT Tower and Projector repository."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils

from vtb.adapters.kimi_k26 import PROJECTOR_PREFIX, SOURCE_REPO, TOWER_PREFIX

SOURCE_MODULE = "modeling_kimi_k25"
SOURCE_REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
WEIGHT_REPO = "exolabs/Kimi-K2.6-vision"
WEIGHT_REVISION = "b20f6d9fcbfef482ef870153073df85f20ddb9e6"
WEIGHT_FILE = "kimi_k26_vision.safetensors"
TOWER_TENSORS = 329
PROJECTOR_TENSORS = 6

BUNDLE_FILES = (
    "config.json",
    "configuration_moonvit.py",
    "modeling_moonvit.py",
    "kimi_k25_vision_processing.py",
    "media_utils.py",
    "preprocessor_config.json",
    "projector_config.json",
    "projector.py",
    "README.md",
    "examples",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
)

CONFIGURATION_SOURCE = '''from transformers.configuration_utils import PretrainedConfig


class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit_k26"

    def __init__(
        self,
        patch_size=14,
        init_pos_emb_height=64,
        init_pos_emb_width=64,
        init_pos_emb_time=4,
        pos_emb_type="divided_fixed",
        num_attention_heads=16,
        num_hidden_layers=27,
        hidden_size=1152,
        intermediate_size=4304,
        merge_kernel_size=(2, 2),
        video_attn_type="spatial_temporal",
        merge_type="sd2_tpool",
        _attn_implementation="eager",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation
'''


def split(weights: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): weight
        for name, weight in weights.items()
        if name.startswith(prefix)
    }


def remote_class(name: str):
    # The K2.6 code targets transformers 4.x and imports one helper that 5.x dropped.
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    return get_class_from_dynamic_module(
        f"{SOURCE_MODULE}.{name}", SOURCE_REPO, revision=SOURCE_REVISION
    )


def load_source_parts(dtype: torch.dtype = torch.bfloat16, attention: str = "eager"):
    settings = json.loads(
        Path(hf_hub_download(SOURCE_REPO, "config.json", revision=SOURCE_REVISION)).read_text()
    )["vision_config"]
    settings["_attn_implementation"] = attention
    source = SimpleNamespace(**settings)

    weights = load_file(hf_hub_download(WEIGHT_REPO, WEIGHT_FILE, revision=WEIGHT_REVISION))
    tower = remote_class("MoonViT3dPretrainedModel")(remote_class("VisionTowerConfig")(source))
    tower.load_state_dict(split(weights, TOWER_PREFIX))
    projector = remote_class("PatchMergerMLP")(remote_class("ProjectorConfig")(source))
    projector.load_state_dict(split(weights, PROJECTOR_PREFIX))
    return tower.to(dtype).eval(), projector.to(dtype).eval()


def standalone_modeling_source(source: str) -> str:
    header_end = source.index("import math")
    body_end = source.index("class KimiK25PreTrainedModel")
    body = source[header_end:body_end]
    body = body.replace("from transformers.cache_utils import Cache\n", "")
    body = body.replace(
        "from transformers.models.llava.modeling_llava import \\\n    LlavaCausalLMOutputWithPast\n",
        "",
    )
    body = body.replace("from .configuration_kimi_k25 import KimiK25Config\n", "")
    body = body.replace("from .modeling_deepseek import DeepseekV3ForCausalLM\n", "")
    return (
        source[:header_end]
        + body
        + "\nfrom .configuration_moonvit import MoonViTConfig\n\n"
        + "class MoonViTModel(MoonViT3dPretrainedModel):\n"
        + "    config_class = MoonViTConfig\n\n"
        + "    def __init__(self, config, *inputs, **kwargs):\n"
        + "        super().__init__(config, *inputs, **kwargs)\n"
        + "        self.post_init()\n"
    )


def standalone_media_utils_source(source: str) -> str:
    source = source.replace(
        "from datetime import datetime, timezone\n",
        "from dataclasses import dataclass\nfrom datetime import datetime, timezone\n",
    )
    source = source.replace("from pydantic import BaseModel, Field\n", "")
    start = source.index("class VideoSpec")
    end = source.index("class ImageInput")
    video_spec = '''@dataclass
class VideoSpec:
    media_type: str
    height: int
    width: int
    num_frames: int
    fps: float
    key_indices: list[int] | None = None
    frame_time_info: dict | None = None


'''
    return source[:start] + video_spec + source[end:]


def standalone_processor_source(source: str) -> str:
    old = """        if not isinstance(medias, list):
            medias = [medias]
        if medias:
"""
    new = """        if not isinstance(medias, list):
            medias = [medias]
        medias = [
            item if isinstance(item, dict) else {"type": "image", "image": item}
            for item in medias
        ]
        if medias:
"""
    if old not in source:
        raise ValueError("Kimi processor input block changed upstream")
    return source.replace(old, new, 1)


def tower_config() -> dict:
    return {
        "architectures": ["MoonViTModel"],
        "auto_map": {
            "AutoConfig": "configuration_moonvit.MoonViTConfig",
            "AutoModel": "modeling_moonvit.MoonViTModel",
        },
        "model_type": "moonvit_k26",
        "patch_size": 14,
        "init_pos_emb_height": 64,
        "init_pos_emb_width": 64,
        "init_pos_emb_time": 4,
        "pos_emb_type": "divided_fixed",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "merge_kernel_size": [2, 2],
        "video_attn_type": "spatial_temporal",
        "merge_type": "sd2_tpool",
        "torch_dtype": "bfloat16",
    }


def projector_config(projector: dict[str, torch.Tensor]) -> dict:
    output_size, hidden_size = projector["proj.2.weight"].shape
    return {
        "merge_kernel_size": [2, 2],
        "input_size": projector["pre_norm.weight"].numel(),
        "hidden_size": hidden_size,
        "output_size": output_size,
        "linear_bias": True,
        "activation_func": "gelu",
        "norm_type": "layernorm",
        "norm_eps": 1e-5,
        "torch_dtype": str(projector["proj.2.weight"].dtype).removeprefix("torch."),
    }


def export(out: Path) -> None:
    weights = load_file(
        hf_hub_download(WEIGHT_REPO, WEIGHT_FILE, revision=WEIGHT_REVISION)
    )
    tower = split(weights, TOWER_PREFIX)
    projector = split(weights, PROJECTOR_PREFIX)
    if len(tower) != TOWER_TENSORS or len(projector) != PROJECTOR_TENSORS:
        raise SystemExit(
            f"expected {TOWER_TENSORS} and {PROJECTOR_TENSORS} tensors, "
            f"read {len(tower)} and {len(projector)}"
        )

    source_files = {
        name: Path(hf_hub_download(SOURCE_REPO, name, revision=SOURCE_REVISION))
        for name in (
            "modeling_kimi_k25.py",
            "kimi_k25_vision_processing.py",
            "media_utils.py",
            "preprocessor_config.json",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
        )
    }

    out.mkdir(parents=True, exist_ok=True)
    save_file(tower, out / "model.safetensors", metadata={"format": "pt"})
    save_file(projector, out / "projector.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(tower_config(), indent=2) + "\n")
    (out / "projector_config.json").write_text(
        json.dumps(projector_config(projector), indent=2) + "\n"
    )
    (out / "configuration_moonvit.py").write_text(CONFIGURATION_SOURCE)
    (out / "modeling_moonvit.py").write_text(
        standalone_modeling_source(source_files["modeling_kimi_k25.py"].read_text())
    )
    (out / "media_utils.py").write_text(
        standalone_media_utils_source(source_files["media_utils.py"].read_text())
    )
    (out / "kimi_k25_vision_processing.py").write_text(
        standalone_processor_source(
            source_files["kimi_k25_vision_processing.py"].read_text()
        )
    )
    processor_config = json.loads(source_files["preprocessor_config.json"].read_text())
    del processor_config["auto_map"]["AutoProcessor"]
    (out / "preprocessor_config.json").write_text(
        json.dumps(processor_config, indent=2) + "\n"
    )
    for name in (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ):
        shutil.copyfile(source_files[name], out / name)

    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/MoonViT-K2.6"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
