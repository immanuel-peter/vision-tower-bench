"""PE-Lang L and the Inkling 2 Towers against references that never run timm's Eva.

PE-Lang L: the weights cell renames Meta's PE-native tensors to timm's names and compares them with the
timm checkpoint the adapter loads. The forward cells run the SHA-pinned perception_models
``core/vision_encoder`` sources on the PE-native tensors.

Inkling 2 Small and Mini: the weights cells compare the release with the pinned parent's ``model.visual.*``
tensors, and ``remap`` covers what parity cannot see. The forward cells run the eager per-image
reference from vLLM's Inkling 2 vision test, with vLLM's own vision classes and weight loader, on
the parent's tensors. Both are compiled from the pinned PR head. The reference only runs in BF16 on CUDA.
"""

import ast
import hashlib
import importlib
import sys
import types
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

from vtb.adapters.pe import (
    DEPTH,
    INKLING2,
    INKLING2_MINI,
    INKLING2_SMALL,
    PE_LANG_L,
    PROJECTOR_TENSORS,
    TOWER_TENSORS,
    VLLM_REFERENCE,
    Inkling2MiniAdapter,
    Inkling2SmallAdapter,
    PELangLAdapter,
    remap_checks,
    source_json,
    source_weights,
    split,
    timm_weights,
)
from vtb.images import square_crop
from vtb.parity import compare, features, fixed_images, weights_match
from vtb.shards import cache_root

REFERENCE_REPO = "facebook/vit_pe_lang_large_patch14_448_timm"
REFERENCE_REVISION = "7eb04e215d0e6e18a21f782a9abc605343dcb503"
REFERENCE_CONFIG = "PE-Lang-L14-448"
UPSTREAM = "facebookresearch/perception_models"
UPSTREAM_COMMIT = "3e352cca660658d4b5c90f42a7808b11469e4c66"
UPSTREAM_SOURCES = {
    "core/vision_encoder/config.py": "37c10ea7b68889241c8d9ebce27c62577c7379a33c444f9b0129bc90946aaff5",
    "core/vision_encoder/pe.py": "36b62204c4cf12a709946facce2df8980e0880bf06deb6e49a2aa288158c3d53",
    "core/vision_encoder/rope.py": "85a16113138f26389ea35fccd313f2aca12945c7ed2eb24e5d52dbe680dad121",
}
VLLM_SOURCES = {
    "vllm/models/inkling2/vision.py": "34525c3b6afb44851de1f0341f60f6006760256c5e507b18cf5167a17497117c",
    "tests/models/inkling2/test_vision.py": "1eddd2ebd456927558a2fd3217335c35d2b3ef2dcbf9af8b76748cf592622f97",
}
VLLM_CLASSES = (
    "VisionMetadata",
    "Inkling2VisionAttention",
    "Inkling2VisionMLP",
    "Inkling2VisionLayerScale",
    "Inkling2VisionLayer",
    "Inkling2VisionEmbeddings",
    "Inkling2VisionBackbone",
    "Inkling2VisionEncoder",
)
VLLM_HELPERS = ("_linear", "_rope", "_normalize", "_reference_image")
VLLM_CONSTANTS = ("_CLIP_MEAN", "_CLIP_STD", "_PROCESSOR_CLIP_STD")
PE_TENSORS = 327
# Relative L2 per cell. PE-Lang L in FP32 on CPU is bit-exact; in BF16 on CUDA it reaches 0.039 at L23,
# where each implementation is already 0.057 from its own FP32 output. Inkling 2 Small against vLLM's
# BF16 reference reaches 0.033 at L23. Inkling 2 Mini shares the Tower and the reference, so it shares the bar.
TOLERANCE = {
    PE_LANG_L.name: {torch.bfloat16: 0.05, torch.float32: 1e-5},
    **{checkpoint.name: {torch.bfloat16: 0.05} for checkpoint in INKLING2},
}
DTYPE = {checkpoint.name: torch.bfloat16 for checkpoint in INKLING2}
REQUIRES_CUDA = {checkpoint.name: True for checkpoint in INKLING2}
INKLING_ADAPTERS = {
    INKLING2_SMALL.name: Inkling2SmallAdapter,
    INKLING2_MINI.name: Inkling2MiniAdapter,
}

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
    return load_file(
        hf_hub_download(
            REFERENCE_REPO, "model.safetensors", revision=REFERENCE_REVISION
        )
    )


def to_timm(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed = {}
    for name, tensor in weights.items():
        for old, new in RENAMES:
            name = name.replace(old, new)
        renamed[name] = (
            tensor.view(TIMM_SHAPES[name]) if name in TIMM_SHAPES else tensor
        )
    return renamed


def pinned_sources(repo: str, commit: str, sources: dict[str, str]) -> Path:
    """Fetch files from a GitHub commit into the shard cache, checked against their SHA-256."""
    root = cache_root().parent / "vtb-upstream" / f"{repo.replace('/', '__')}@{commit}"
    for name, digest in sources.items():
        path = root / name
        if not path.is_file():
            response = requests.get(
                f"https://raw.githubusercontent.com/{repo}/{commit}/{name}", timeout=60
            )
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path} does not match {repo}@{commit}")
    return root


def upstream_pe():
    """Import ``core.vision_encoder.pe`` from the pinned sources without installing perception_models."""
    root = pinned_sources(UPSTREAM, UPSTREAM_COMMIT, UPSTREAM_SOURCES)
    for name, path in (
        ("core", root / "core"),
        ("core.vision_encoder", root / "core" / "vision_encoder"),
    ):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
        elif list(module.__path__) != [str(path)]:
            raise ImportError(f"another {name} package is already imported")
    return importlib.import_module("core.vision_encoder.pe")


def reference_tower(
    weights: dict[str, torch.Tensor], device: str, dtype: torch.dtype
) -> torch.nn.Module:
    model = upstream_pe().VisionTransformer.from_config(REFERENCE_CONFIG)
    model.load_state_dict(weights, strict=True)
    return model.to(device, dtype).eval()


@torch.inference_mode()
def reference_features(
    model: torch.nn.Module, pixel_values: torch.Tensor, layers: list[int]
) -> dict[str, torch.Tensor]:
    captured = {}
    handles = [
        model.transformer.resblocks[layer - 1].register_forward_hook(
            lambda _module, _args, output, layer=layer: captured.update(
                {layer: output[:, 1:].cpu()}
            )
        )
        for layer in layers
    ]
    try:
        model.forward_features(pixel_values, norm=False)
    finally:
        for handle in handles:
            handle.remove()
    return {f"tower:{layer}": captured[layer] for layer in layers}


def validate(
    checkpoint=PE_LANG_L, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> dict:
    if checkpoint in INKLING2:
        return validate_inkling(checkpoint, device, dtype)
    native = reference_weights()
    report = {
        "model": checkpoint.name,
        "source": f"{checkpoint.repo}@{checkpoint.revision}",
        "reference": f"{REFERENCE_REPO}@{REFERENCE_REVISION} through {UPSTREAM}@{UPSTREAM_COMMIT} VisionTransformer('{REFERENCE_CONFIG}')",
        "weights": {
            "tower": weights_match(
                timm_weights(checkpoint), to_timm(native), PE_TENSORS
            )
        },
    }
    adapter = PELangLAdapter(device=device, dtype=dtype)
    images = fixed_images()
    actual = features(adapter, images)
    pixels = adapter.collate([adapter.preprocess()(image) for image in images])[
        "pixel_values"
    ]
    expected = reference_features(
        reference_tower(native, device, dtype),
        pixels.to(device, dtype),
        adapter.depth_points(),
    )
    report["forward"] = {key: compare(actual[key], expected[key]) for key in expected}
    tolerance = TOLERANCE[checkpoint.name][dtype]
    passed = report["weights"]["tower"]["bit_exact"] and all(
        cell["relative_l2"] <= tolerance for cell in report["forward"].values()
    )
    return {**report, "tolerance": tolerance, "passed": passed}


def vllm_reference() -> dict:
    """vLLM's Inkling 2 vision classes and eager test reference, compiled from the pinned PR head without importing vLLM."""
    repo, _, head = VLLM_REFERENCE
    root = pinned_sources(repo, head, VLLM_SOURCES)

    def default_weight_loader(param: torch.Tensor, weight: torch.Tensor) -> None:
        assert param.size() == weight.size(), (param.size(), weight.size())
        param.data.copy_(weight)

    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "Counter": Counter,
        "Iterable": Iterable,
        "Mapping": Mapping,
        "Any": Any,
        "NamedTuple": NamedTuple,
        "default_weight_loader": default_weight_loader,
    }
    wanted = set(VLLM_CLASSES) | set(VLLM_HELPERS) | set(VLLM_CONSTANTS)
    found = set()
    for name in VLLM_SOURCES:
        path = root / name
        nodes = []
        for node in ast.parse(path.read_text()).body:
            names = {
                target.id
                for target in getattr(node, "targets", [])
                if isinstance(target, ast.Name)
            } | {getattr(node, "name", None)}
            if names & wanted:
                nodes.append(node)
                found |= names & wanted
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    if found != wanted:
        raise ValueError(f"pinned vLLM sources lack {sorted(wanted - found)}")
    return namespace


def vllm_encoder(
    namespace: dict, weights: dict[str, torch.Tensor], config: dict, text_hidden: int
) -> nn.Module:
    """The vLLM encoder loaded through its own ``load_weights``, plus one zero-LayerScale layer after the last."""
    namespace["CONFIG"] = config
    encoder = namespace["Inkling2VisionEncoder"](config, text_hidden_size=text_hidden)
    loaded = encoder.load_weights(weights.items())
    if loaded != set(dict(encoder.named_parameters())):
        raise ValueError(
            f"vLLM load_weights left {sorted(set(dict(encoder.named_parameters())) - loaded)[:5]} unset"
        )
    # The reference keeps the residual stream inside one loop, so read each layer's output as the next
    # layer's ln_1 input. The extra layer adds zero-scaled updates, so the stream it passes on is unchanged.
    sentinel = namespace["Inkling2VisionLayer"](config)
    for name, parameter in sentinel.named_parameters():
        parameter.data.fill_(
            1.0 if name.endswith(("ln_1.weight", "ln_2.weight")) else 0.0
        )
    encoder.backbone.layers.append(sentinel)
    return encoder.cuda().eval()


@torch.inference_mode()
def vllm_features(
    namespace: dict, encoder: nn.Module, image, resolution: int, layers: list[int]
) -> dict[str, torch.Tensor]:
    """One image through vLLM's ``_reference_image`` as 40 x 40 raw-pixel blocks in raster order."""
    side = encoder.patch_size * encoder.merge_size
    blocks = resolution // side
    raw = torch.from_numpy(np.array(square_crop(resolution)(image.convert("RGB"))))
    raw = (
        raw.reshape(blocks, side, blocks, side, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(-1, side, side, 3)
        .cuda()
        .bfloat16()
    )
    captured, handles = {}, []
    for layer in layers:
        handles.append(
            encoder.backbone.layers[layer].ln_1.register_forward_pre_hook(
                lambda _module, args, layer=layer: captured.update(
                    {f"tower:{layer}": args[0][1:].cpu()}
                )
            )
        )
    handles.append(
        encoder.proj_in.register_forward_pre_hook(
            lambda _module, args: captured.update({f"merged:{DEPTH}": args[0].cpu()})
        )
    )
    try:
        captured[f"projected:{DEPTH}"] = namespace["_reference_image"](
            encoder, raw, blocks, blocks
        ).cpu()
    finally:
        for handle in handles:
            handle.remove()
    pixels = (
        namespace["_normalize"](raw)
        .reshape(blocks, blocks, side, side, 3)
        .permute(4, 0, 2, 1, 3)
        .reshape(3, resolution, resolution)
    )
    return {**captured, "pixels": pixels.float().cpu()}


def validate_inkling(
    checkpoint=INKLING2_SMALL, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> dict:
    if not str(device).startswith("cuda") or dtype != torch.bfloat16:
        raise ValueError("the vLLM reference runs in BF16 on CUDA")
    parent = source_weights(checkpoint)
    tower, projector = split(parent)
    bundle = checkpoint.open()
    repo, number, head = VLLM_REFERENCE
    report = {
        "model": checkpoint.name,
        "source": f"{checkpoint.repo}@{checkpoint.revision}",
        "reference": f"eager _reference_image from https://github.com/{repo}/pull/{number} at {head}, on the parent's tensors",
        "weights": {
            "tower": weights_match(
                load_file(bundle / "model.safetensors"), tower, TOWER_TENSORS
            ),
            "projector": weights_match(
                load_file(bundle / "projector.safetensors"),
                projector,
                PROJECTOR_TENSORS,
            ),
        },
        "remap": remap_checks(checkpoint, parent),
    }
    config = source_json(checkpoint, "config.json")
    namespace = vllm_reference()
    encoder = vllm_encoder(
        namespace, parent, config["vision_config"], config["text_config"]["hidden_size"]
    )
    del parent, tower, projector
    adapter = INKLING_ADAPTERS[checkpoint.name](device=device, dtype=dtype)
    images = fixed_images()
    actual = features(adapter, images)
    references = [
        vllm_features(
            namespace, encoder, image, adapter.resolution, adapter.depth_points()
        )
        for image in images
    ]
    expected = {
        key: torch.stack([reference[key] for reference in references])
        for key in references[0]
    }
    pixels = torch.stack([adapter.preprocess()(image) for image in images])
    report["preprocessing"] = compare(pixels, expected.pop("pixels"))
    report["forward"] = {key: compare(actual[key], expected[key]) for key in expected}
    grid = adapter.resolution // checkpoint.patch_size
    tokens_ok = all(
        cell["shape"][1] == (grid**2 if key.startswith("tower") else (grid // 2) ** 2)
        for key, cell in report["forward"].items()
    )
    tolerance = TOLERANCE[checkpoint.name][dtype]
    passed = (
        all(cell["bit_exact"] for cell in report["weights"].values())
        and report["remap"]["passed"]
        and tokens_ok
        and report["preprocessing"]["bit_exact"]
        and all(cell["relative_l2"] <= tolerance for cell in report["forward"].values())
    )
    return {
        **report,
        "resolution": adapter.resolution,
        "tolerance": tolerance,
        "passed": passed,
    }
