"""PE-Lang L against Meta's PE-native checkpoint, run through the perception_models VisionTransformer rather than timm's Eva.

The weights cell renames the PE-native tensors to timm's names and compares them with the timm release
the adapter loads. The forward cells run the SHA-pinned upstream ``core/vision_encoder`` sources on the
PE-native tensors and compare every sampled block's patch tokens with the adapter's.
"""

import hashlib
import importlib
import sys
import types
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vtb.adapters.pe import DEPTH, PE_LANG_L, PELangLAdapter, timm_weights
from vtb.parity import compare, features, fixed_images, weights_match
from vtb.shards import cache_root

REFERENCE_REPO = "facebook/vit_pe_lang_large_patch14_448_timm"
REFERENCE_REVISION = "7eb04e215d0e6e18a21f782a9abc605343dcb503"
REFERENCE_CONFIG = "PE-Lang-L14-448"
UPSTREAM = "facebookresearch/perception_models"
UPSTREAM_COMMIT = "3e352cca660658d4b5c90f42a7808b11469e4c66"
UPSTREAM_SOURCES = {
    "config.py": "37c10ea7b68889241c8d9ebce27c62577c7379a33c444f9b0129bc90946aaff5",
    "pe.py": "36b62204c4cf12a709946facce2df8980e0880bf06deb6e49a2aa288158c3d53",
    "rope.py": "85a16113138f26389ea35fccd313f2aca12945c7ed2eb24e5d52dbe680dad121",
}
TENSORS = 327
# Relative L2 per cell. FP32 on CPU is bit-exact; BF16 on CUDA reaches 0.039 at L23, where each
# implementation is already 0.057 from its own FP32 output.
TOLERANCE = {torch.bfloat16: 0.05, torch.float32: 1e-5}
ADAPTERS = {PE_LANG_L.name: PELangLAdapter}

RENAMES = (
    ("class_embedding", "cls_token"),
    ("positional_embedding", "pos_embed"),
    ("conv1.", "patch_embed.proj."),
    ("ln_pre.", "norm_pre."),
    ("transformer.resblocks.", "blocks."),
    ("attn.in_proj_", "attn.qkv."),
    ("attn.out_proj.", "attn.proj."),
    ("ln_1.", "norm1."),
    ("ln_2.", "norm2."),
    ("ls_1.gamma", "gamma_1"),
    ("ls_2.gamma", "gamma_2"),
    ("mlp.c_fc.", "mlp.fc1."),
    ("mlp.c_proj.", "mlp.fc2."),
)
TIMM_SHAPES = {"cls_token": (1, 1, -1), "pos_embed": (1, -1, 1024)}


def reference_weights() -> dict[str, torch.Tensor]:
    return load_file(hf_hub_download(REFERENCE_REPO, "model.safetensors", revision=REFERENCE_REVISION))


def to_timm(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed = {}
    for name, tensor in weights.items():
        for old, new in RENAMES:
            name = name.replace(old, new)
        renamed[name] = tensor.view(TIMM_SHAPES[name]) if name in TIMM_SHAPES else tensor
    return renamed


def upstream_sources() -> Path:
    root = cache_root().parent / "vtb-upstream" / f"perception_models@{UPSTREAM_COMMIT}"
    package = root / "core" / "vision_encoder"
    package.mkdir(parents=True, exist_ok=True)
    for name, digest in UPSTREAM_SOURCES.items():
        path = package / name
        if not path.is_file():
            url = f"https://raw.githubusercontent.com/{UPSTREAM}/{UPSTREAM_COMMIT}/core/vision_encoder/{name}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path} does not match {UPSTREAM}@{UPSTREAM_COMMIT}")
    return root


def upstream_pe():
    """Import ``core.vision_encoder.pe`` from the pinned sources without installing perception_models."""
    root = upstream_sources()
    for name, path in (("core", root / "core"), ("core.vision_encoder", root / "core" / "vision_encoder")):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
        elif list(module.__path__) != [str(path)]:
            raise ImportError(f"another {name} package is already imported")
    return importlib.import_module("core.vision_encoder.pe")


def reference_tower(weights: dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> torch.nn.Module:
    model = upstream_pe().VisionTransformer.from_config(REFERENCE_CONFIG)
    model.load_state_dict(weights, strict=True)
    if model.layers != DEPTH or model.pool_type != "none" or not isinstance(model.ln_post, torch.nn.Identity):
        raise ValueError(f"{REFERENCE_CONFIG} is not the {DEPTH}-block PE-Lang L without ln_post or pooling")
    return model.to(device, dtype).eval()


@torch.inference_mode()
def reference_features(model: torch.nn.Module, pixel_values: torch.Tensor, layers: list[int]) -> dict[str, torch.Tensor]:
    captured = {}
    handles = [model.transformer.resblocks[layer - 1].register_forward_hook(
        lambda _module, _args, output, layer=layer: captured.update({layer: output[:, 1:].cpu()})
    ) for layer in layers]
    try:
        model.forward_features(pixel_values, norm=False)
    finally:
        for handle in handles:
            handle.remove()
    return {f"tower:{layer}": captured[layer] for layer in layers}


def validate(checkpoint=PE_LANG_L, device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    native = reference_weights()
    report = {
        "model": checkpoint.name, "source": f"{checkpoint.repo}@{checkpoint.revision}",
        "reference": f"{REFERENCE_REPO}@{REFERENCE_REVISION} through {UPSTREAM}@{UPSTREAM_COMMIT} VisionTransformer('{REFERENCE_CONFIG}')",
        "weights": {"tower": weights_match(timm_weights(checkpoint), to_timm(native), TENSORS)},
    }
    adapter = ADAPTERS[checkpoint.name](device=device, dtype=dtype)
    images = fixed_images()
    actual = features(adapter, images)
    pixels = adapter.collate([adapter.preprocess()(image) for image in images])["pixel_values"]
    expected = reference_features(reference_tower(native, device, dtype), pixels.to(device, dtype), adapter.depth_points())
    report["forward"] = {key: compare(actual[key], expected[key]) for key in expected}
    tolerance = TOLERANCE[dtype]
    passed = report["weights"]["tower"]["bit_exact"] and all(cell["relative_l2"] <= tolerance for cell in report["forward"].values())
    return {**report, "tolerance": tolerance, "passed": passed}
