"""Compare a MiMo 2.6 release with the pinned parent's own vision class.

The reference imports MiMoVisionTransformer from the SHA-pinned parent remote
code and never constructs the parent language model. That class cannot strictly
load its own checkpoint: it creates three merger biases the checkpoint lacks.
The reference sets only those biases to zero and leaves the forward untouched.
"""

import gc
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vtb.adapters.mimo26 import FLASH, PRO, VISUAL_TENSORS, MiMo26FlashAdapter, MiMo26ProAdapter, source_weights
from vtb.images import square_crop
from vtb.parity import compare, reference_raster, weights_match

ADAPTERS = {PRO.name: MiMo26ProAdapter, FLASH.name: MiMo26FlashAdapter}
ABSENT_BIASES = {"merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"}
TOLERANCE = {torch.bfloat16: 0.016, torch.float32: 1e-4}


def load_reference(checkpoint, weights, device, dtype):
    config = AutoConfig.from_pretrained(
        checkpoint.repo, revision=checkpoint.revision, code_revision=checkpoint.revision, trust_remote_code=True,
    )
    cls = get_class_from_dynamic_module(
        "modeling_mimo_v2.MiMoVisionTransformer", checkpoint.repo,
        revision=checkpoint.revision, code_revision=checkpoint.revision,
    )
    with torch.device("meta"):
        parent = cls(SimpleNamespace(**config.vision_config))
    loaded = parent.load_state_dict(weights, strict=False, assign=True)
    assert set(loaded.missing_keys) == ABSENT_BIASES and not loaded.unexpected_keys, loaded
    for name in ABSENT_BIASES:
        module_name, parameter = name.rsplit(".", 1)
        module = parent.get_submodule(module_name)
        setattr(module, parameter, torch.nn.Parameter(torch.zeros(getattr(module, parameter).shape, dtype=torch.bfloat16)))
    # The nonpersistent RoPE buffer is still on the meta device. HF keeps these
    # frequencies in float32 under BF16 weights, so set them again after casting.
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, 32, 2).float() / 32))
    parent.rotary_pos_emb.inv_freq = frequencies
    parent = parent.to(device=device, dtype=dtype).eval()
    parent.rotary_pos_emb.inv_freq = frequencies.to(device)
    return config, cls, parent


def validate(checkpoint, device="cpu", dtype=torch.float32, resolutions=(96, 448), bundle=None):
    name, bundle = checkpoint.name, bundle or checkpoint.bundle
    weights = source_weights(checkpoint)
    identity = {"tower": weights_match(load_file(bundle / "model.safetensors"), {"visual." + key: value for key, value in weights.items()}, VISUAL_TENSORS)}
    config, cls, parent = load_reference(checkpoint, weights, device, dtype)
    adapter = ADAPTERS[name](resolution=resolutions[0], device=device, dtype=dtype, bundle=bundle)
    del weights
    report = {
        "model": name, "source": checkpoint.repo, "revision": checkpoint.revision,
        "torch": torch.__version__, "device": str(device), "dtype": str(dtype), "weights": identity, "weights_bit_exact": identity["tower"]["bit_exact"],
        "parent_source_sha256": hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest(),
        "reference": "pinned parent MiMoVisionTransformer; only the three absent merger biases set to zero",
        "zero_completed_biases": sorted(ABSENT_BIASES), "cases": [],
    }
    processor = AutoImageProcessor.from_pretrained(checkpoint.repo, revision=checkpoint.revision, trust_remote_code=True)
    rng = np.random.default_rng(726)
    images = [Image.fromarray(rng.integers(0, 256, (317, 503, 3), dtype=np.uint8)),
              Image.fromarray(rng.integers(0, 256, (401, 299, 3), dtype=np.uint8))]
    released = AutoModel.from_pretrained(bundle, local_files_only=True, trust_remote_code=True, dtype=dtype).to(device).eval()
    for resolution in resolutions:
        adapter.resolution = resolution
        inputs = adapter.collate([adapter.preprocess()(image) for image in images])
        processed = processor(images=[square_crop(resolution)(image) for image in images], do_resize=False, return_tensors="pt")
        torch.testing.assert_close(inputs["pixel_values"], processed["pixel_values"], rtol=0, atol=5e-7)
        assert torch.equal(inputs["grid_thw"], processed["image_grid_thw"])
        captured, handles = {}, []

        def capture(key):
            def hook(_module, _args, output):
                captured[key] = output.detach()
            return hook

        for layer in adapter.depth_points():
            handles.append(parent.blocks[layer - 1].register_forward_hook(capture(layer)))
        handles.append(parent.merger.ln_q.register_forward_hook(capture("merged")))
        try:
            with torch.inference_mode():
                projected = parent(inputs["pixel_values"].to(device, dtype), inputs["grid_thw"])
        finally:
            for handle in handles:
                handle.remove()
        case = {"resolution": resolution, "batch_size": len(images),
                "preprocessing": compare(inputs["pixel_values"], processed["pixel_values"]), "features": []}
        for batch in adapter.extract(inputs, ["noise_wide", "noise_tall"]):
            if batch.stage == "tower":
                column_major = parent.vit_window_attn_types[batch.layer_index - 1] == 1
                expected = reference_raster(captured[batch.layer_index], inputs["grid_thw"], column_major)
            elif batch.stage == "merged":
                expected = captured["merged"].reshape(len(images), -1, 5120)
            else:
                expected = projected.reshape(len(images), -1, config.vision_config["out_hidden_size"])
            case["features"].append({"stage": batch.stage, "layer": batch.layer_index, **compare(batch.tokens, expected.cpu())})
        with torch.inference_mode():
            actual = released(inputs["pixel_values"].to(device, dtype), inputs["grid_thw"])
            rms_merged = torch.nn.functional.rms_norm(captured[28], (1280,), parent.merger.ln_q.weight, eps=1e-6).reshape(-1, 5120)
            rms_projected = parent.merger.mlp(rms_merged)
        case["auto_model_projected"] = compare(actual.cpu(), projected.cpu())
        # vLLM feeds this merger RMSNorm instead of LayerNorm. Replace only that.
        case["rmsnorm_merger_ablation"] = compare(rms_projected.cpu(), projected.cpu())
        report["cases"].append(case)
    del adapter, parent, released
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**report, "passed": passes(report)}


def passes(report: dict) -> bool:
    tolerance = TOLERANCE[torch.bfloat16 if report["dtype"] == str(torch.bfloat16) else torch.float32]
    cells = [cell for case in report["cases"] for cell in (*case["features"], case["auto_model_projected"])]
    return report["weights_bit_exact"] and all(cell["max_abs"] <= tolerance for cell in cells)
