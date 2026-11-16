from collections.abc import Iterator
from functools import partial
from json import loads
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, PatchPreprocess, default_device, depth_points, make_batch, pack_patches
from vtb.bundle import expect_tensors, write_bundle
from vtb.feature_batch import FeatureBatch
from vtb.provenance import log_loaded

MODEL_ID = "immanuelpeter/MoonViT-K2.6"
PROJECTOR_FILE = "projector.safetensors"
PROJECTOR_CONFIG = "projector_config.json"

# Kimi K2.6 is where export reads both halves from.
SOURCE_REPO = "moonshotai/Kimi-K2.6"
SOURCE_SHARDS = ("model-00063-of-000064.safetensors", "model-00064-of-000064.safetensors")
TOWER_PREFIX = "vision_tower."
PROJECTOR_PREFIX = "mm_projector."
PATCH_SIZE = 14


Preprocess = partial(PatchPreprocess, mean=HALF, std=HALF, patch_size=PATCH_SIZE, flatten=False)
collate = partial(pack_patches, grid_key="grid_thws")


def load_projector_weights(dtype: torch.dtype) -> nn.Module:
    """Build Kimi K2.6's patch merger from its published weights."""
    settings = loads(Path(hf_hub_download(MODEL_ID, PROJECTOR_CONFIG)).read_text())
    projector = nn.Module()
    projector.pre_norm = nn.LayerNorm(settings["input_size"], eps=settings["norm_eps"])
    projector.proj = nn.Sequential(
        nn.Linear(settings["hidden_size"], settings["hidden_size"]),
        nn.GELU(),
        nn.Linear(settings["hidden_size"], settings["output_size"]),
    )
    projector.load_state_dict(load_file(hf_hub_download(MODEL_ID, PROJECTOR_FILE)))
    return projector.to(dtype).eval()


def load_parts(dtype: torch.dtype, attention: str = "eager"):
    log_loaded(MODEL_ID, MODEL_ID)
    tower = AutoModel.from_pretrained(
        MODEL_ID, dtype=dtype, trust_remote_code=True, attn_implementation=attention
    ).eval()
    return tower, load_projector_weights(dtype)


class KimiK26Adapter:
    """Use batch size 1 without flash attention, as measured."""

    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        tower, projector = load_parts(dtype)
        self.model = tower.to(self.device)
        self.projector = projector.to(self.device)
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        points = self.depth_points()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture(layer: int):
            def hook(_module, _args, output):
                captured[layer] = output.detach()

            return hook

        for layer in points:
            # Use the encoder output at the final layer to retain its norm.
            target = self.model.encoder if layer == self.num_layers else self.model.encoder.blocks[layer - 1]
            handles.append(target.register_forward_hook(capture(layer)))

        try:
            merged = self.model(
                inputs["pixel_values"].to(self.device, self.dtype),
                inputs["grid_thws"].to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size
        for layer in points:
            yield make_batch(self, captured[layer].view(rows, -1, width), image_ids, "tower", layer)

        stacked = torch.stack(merged)
        yield make_batch(self, stacked.flatten(2), image_ids, "merged", self.num_layers)
        grouped = self.projector.pre_norm(stacked).flatten(2)
        yield make_batch(self, self.projector.proj(grouped), image_ids, "projected", self.num_layers)


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
    settings = loads(
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
    weights = load_file(hf_hub_download(WEIGHT_REPO, WEIGHT_FILE, revision=WEIGHT_REVISION))
    tower = expect_tensors(split(weights, TOWER_PREFIX), TOWER_TENSORS, "Tower")
    projector = expect_tensors(split(weights, PROJECTOR_PREFIX), PROJECTOR_TENSORS, "Projector")

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
    processor_config = loads(source_files["preprocessor_config.json"].read_text())
    del processor_config["auto_map"]["AutoProcessor"]

    write_bundle(
        out,
        weights={"model.safetensors": tower, "projector.safetensors": projector},
        configs={
            "config.json": tower_config(),
            "projector_config.json": projector_config(projector),
            "preprocessor_config.json": processor_config,
        },
        license=source_files["LICENSE"],
        files={"THIRD_PARTY_NOTICES.md": source_files["THIRD_PARTY_NOTICES.md"]},
    )
    (out / "configuration_moonvit.py").write_text(CONFIGURATION_SOURCE)
    (out / "modeling_moonvit.py").write_text(standalone_modeling_source(source_files["modeling_kimi_k25.py"].read_text()))
    (out / "media_utils.py").write_text(standalone_media_utils_source(source_files["media_utils.py"].read_text()))
    (out / "kimi_k25_vision_processing.py").write_text(
        standalone_processor_source(source_files["kimi_k25_vision_processing.py"].read_text())
    )
