"""Independent pinned-parent BF16 parity. Never constructs a parent LLM.

The reference uses the parent checkpoint's RadioModel and VisionProjector code,
not build_parts or the local release implementation. Explicit key conversion
reproduces the conversion rules shipped in the pinned parent modeling_radio.py.
"""

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.integrations import use_kernel_forward_from_hub

from vtb.adapters.nemotron35 import (
    BOLT, CHECKPOINTS, SUPER, Nemotron35BoltAdapter, Nemotron35SuperAdapter, Preprocess, source_weights,
)

ADAPTERS = {SUPER.name: Nemotron35SuperAdapter, BOLT.name: Nemotron35BoltAdapter}
TOLERANCE = {torch.bfloat16: (1e-3, 0.999999), torch.float32: (1e-3, 0.99999)}


def convert_tower(weights):
    result = {}
    replacements = (
        ("radio_model.model.patch_generator.video_embedder", "embeddings.video_patch_projection"),
        ("radio_model.model.patch_generator.embedder", "embeddings.patch_projection"),
        ("radio_model.model.patch_generator.pos_embed", "embeddings.position_embedding"),
        ("radio_model.model.patch_generator.cls_token.token", "embeddings.cls_register_token"),
        ("radio_model.model.blocks", "encoder.layer"),
        ("attn.proj", "attention.output.dense"),
        ("radio_model.input_conditioner", "input_conditioner"),
    )
    for key, value in weights.items():
        for old, new in replacements:
            key = key.replace(old, new)
        if ".attn.qkv." in key:
            for kind, tensor in zip(("query", "key", "value"), value.chunk(3, dim=0), strict=True):
                result[key.replace("attn.qkv", f"attention.attention.{kind}")] = tensor
        else:
            result[key] = value
    return result


def parent_projector_class(checkpoint):
    # Bolt's top-level module imports mamba_ssm for its unused LLM. Compile the
    # three unmodified vision class ASTs from the pinned file instead. This
    # isolates actual parent code without stubbing or implementing its math.
    path = Path(hf_hub_download(checkpoint.repo, "modeling_nemotron_h_omni.py", revision=checkpoint.revision))
    names = {"NemotronH_Omni_RMSNorm", "NemotronH_Omni_Reasoning_V3MLP", "NemotronH_Omni_Reasoning_V3VisionProjector"}
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == 3
    import warnings
    namespace = {"torch": torch, "nn": torch.nn, "warnings": warnings, "ACT2FN": ACT2FN,
                 "use_kernel_forward_from_hub": use_kernel_forward_from_hub}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["NemotronH_Omni_Reasoning_V3VisionProjector"]


def load_reference(checkpoint, dtype, device):
    config = AutoConfig.from_pretrained(checkpoint.repo, revision=checkpoint.revision, trust_remote_code=True)
    cls = get_class_from_dynamic_module("modeling_radio.RadioModel", checkpoint.repo, revision=checkpoint.revision)
    projector_cls = parent_projector_class(checkpoint)
    parts = source_weights(checkpoint)
    radio = cls(config.vision_config)
    missing, unexpected = radio.load_state_dict(convert_tower(parts["tower"]), strict=False)
    allowed = {f"encoder.layer.{i}.layer_scale{j}.lambda1" for i in range(32) for j in (1, 2)}
    assert set(missing) == allowed and not unexpected, (missing, unexpected)
    assert all(torch.equal(radio.state_dict()[key], torch.ones_like(radio.state_dict()[key])) for key in allowed)
    # The parent's composition applies its FP32 processor externally.
    conditioner = radio.make_preprocessor_external()
    radio.to(device, dtype).eval()
    projector = projector_cls(config)
    mapped = {f"mlp1.{new}.weight": parts["projector"][f"{old}.weight"] for old, new in (("0", "norm"), ("1", "linear1"), ("3", "linear2"))}
    mapped.update({f"vision_final_layernorm.{key}": value for key, value in parts["final_norm"].items()})
    projector.load_state_dict(mapped, strict=True)
    assert (projector.vision_final_layernorm is not None) == checkpoint.final_layernorm
    return radio, projector.to(device, dtype).eval(), conditioner.to(device).float()


def metrics(actual, expected):
    a, e = actual.cpu().double().flatten(), expected.cpu().double().flatten()
    return {
        "shape": list(actual.shape), "bit_exact": torch.equal(actual, expected),
        "max_abs": (a - e).abs().max().item(),
        "relative_l2": ((a - e).norm() / e.norm().clamp_min(1e-12)).item(),
        "cosine": max(-1., min(1., F.cosine_similarity(a[None], e[None]).item())),
    }


def validate(name, bundle, device, resolution=448, dtype=torch.bfloat16, image_path=None):
    checkpoint = CHECKPOINTS[name]
    weights = source_weights(checkpoint)
    for part, filename in (("tower", "model.safetensors"), ("projector", "projector.safetensors"), ("final_norm", "final_norm.safetensors")):
        actual = load_file(bundle / filename) if weights[part] else {}
        assert set(actual) == set(weights[part])
        for key, value in actual.items():
            assert value.dtype == weights[part][key].dtype and torch.equal(value, weights[part][key]), key
    y, x = np.mgrid[:480, :640]
    image = Image.fromarray(np.stack((x % 256, y % 256, ((x // 7 + y // 9) % 2) * 255), axis=-1).astype(np.uint8))
    samples = [Preprocess(resolution)(image)]
    if image_path is not None:
        with Image.open(image_path) as raw:
            samples.append(Preprocess(resolution)(raw.convert("RGB")))
    pixels = torch.stack(samples).to(device)
    ref, projector, conditioner = load_reference(checkpoint, dtype, device)
    capture = {}
    handles = [ref.encoder.layer[layer - 1].register_forward_hook(
        lambda _m, _i, output, layer=layer: capture.update({layer: output.detach()})
    ) for layer in range(4, 33, 4)]
    with torch.inference_mode():
        normalized = conditioner(pixels).to(dtype)
        ref_output = ref(normalized)
        expected_projected = projector(normalized, ref)
        normed = ref_output.features if projector.vision_final_layernorm is None else projector.vision_final_layernorm(ref_output.features)
        merged = projector.pixel_shuffle(normed.reshape(len(pixels), resolution // 16, resolution // 16, 1280), 0.5).flatten(1, 2)
        expected = {("tower", layer): tensor[:, 10:].cpu() for layer, tensor in capture.items()}
        expected[("merged", 32)] = merged.cpu()
        expected[("projected", 32)] = expected_projected.cpu()
    for handle in handles:
        handle.remove()
    del ref, projector, conditioner, ref_output, capture
    gc.collect()
    torch.cuda.empty_cache()
    os.environ[f"VTB_{name.upper()}_BUNDLE"] = str(bundle)
    adapter = ADAPTERS[name](resolution=resolution, device=device, dtype=dtype, stages=("tower", "merged", "projected"))
    report = {
        "model": name, "source": checkpoint.repo, "revision": checkpoint.revision,
        "torch": torch.__version__, "device": str(device), "dtype": str(dtype), "resolution": resolution,
        "images": ["nonuniform synthetic raster"] + ([str(image_path)] if image_path else []),
        "raster_sha256": hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest(),
        "weights_bit_exact": True, "tensor_counts": {key: len(value) for key, value in weights.items()},
        "reference": "pinned parent RadioModel and unmodified VisionProjector class ASTs; explicit checkpoint conversion; no parent LLM",
        "reference_code_sha256": {
            filename: hashlib.sha256(Path(hf_hub_download(checkpoint.repo, filename, revision=checkpoint.revision)).read_bytes()).hexdigest()
            for filename in ("modeling_radio.py", "modeling_nemotron_h_omni.py", "configuration_radio.py")
        },
        "cells": {},
    }
    for batch in adapter.extract({"pixel_values": pixels}, [str(i) for i in range(len(pixels))]):
        key = (batch.stage, batch.layer_index)
        report["cells"][f"{key[0]}:{key[1]}"] = metrics(batch.tokens, expected[key])
    # Test the delivered source as well, always against the independent parent.
    spec = importlib.util.spec_from_file_location("release_vision", bundle / "vision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    released = module.load_vision(bundle, dtype=dtype).to(device)
    output = released(pixels)
    report["release_cells"] = {stage: metrics(value.cpu(), expected[(stage, 32)]) for stage, value in output.items()}
    return report


def assert_parity(report, dtype):
    relative_l2, cosine = TOLERANCE[dtype]
    cells = [*report["cells"].values(), *report["release_cells"].values()]
    assert all(cell["relative_l2"] < relative_l2 and cell["cosine"] > cosine for cell in cells), "parent forward parity failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, required=True)
    parser.add_argument("--bundle", type=Path, default=None, help="defaults to the tracked hf/ release directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--image", type=Path, help="include this natural image alongside the synthetic raster")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    bundle = args.bundle or CHECKPOINTS[args.model].release_dir
    report = validate(args.model, bundle, torch.device(args.device), args.resolution, getattr(torch, args.dtype), args.image)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert_parity(report, getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
