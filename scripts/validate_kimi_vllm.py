"""Compare each exported Kimi checkpoint to its official vLLM vision path."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from scripts.kimi_vllm_reference import load_reference
from vtb.adapters.moonvit_v2 import CHECKPOINTS, MoonViTV2Adapter, collate, load_projector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa", "sdpa_math", "official_sdpa"), default="official_sdpa")
    parser.add_argument("--rotary", choices=("auto", "native", "official_packed"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--diagnostic", action="store_true", help="retain all numerical failures without aborting the grid")
    parser.add_argument("--isolate", action="store_true", help="record stem, first block and rotary kernel isolation")
    args = parser.parse_args()
    if args.rotary == "auto":
        args.rotary = "official_packed" if args.device.startswith("cuda") else "native"
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    checkpoint = CHECKPOINTS[args.checkpoint]
    parent = json.loads(args.parent_config.read_text())
    cfg = dict(parent["vision_config"])
    for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers"):
        cfg[key] = cfg["vt_" + key]
    cfg["hidden_act"] = cfg["activation_func"]
    reference = load_reference(args.source, args.attention, args.rotary)
    config = SimpleNamespace(**cfg)
    index_path = hf_hub_download(checkpoint.repo, "model.safetensors.index.json",
                                 revision=checkpoint.revision, local_files_only=True)
    index = json.loads(Path(index_path).read_text())["weight_map"]
    parent_weights = {}
    for filename, prefix in (("model.safetensors", "vision_tower."),
                             ("projector.safetensors", "mm_projector.")):
        inside = {}
        for shard in sorted({shard for name, shard in index.items() if name.startswith(prefix)}):
            path = hf_hub_download(checkpoint.repo, shard, revision=checkpoint.revision, local_files_only=True)
            inside.update({name.removeprefix(prefix): tensor for name, tensor in load_file(path).items()
                           if name.startswith(prefix)})
        exported = load_file(args.bundle / filename)
        assert inside.keys() == exported.keys()
        assert all(torch.equal(inside[name], exported[name]) for name in inside)
        parent_weights[filename] = inside
    tower = reference.MoonViT3dPretrainedModel(config).to(dtype=dtype, device=args.device).eval()
    tower.load_state_dict(parent_weights["model.safetensors"], strict=True)
    projector = reference.KimiK25MultiModalProjector(config, cfg["text_hidden_size"]).to(dtype=dtype, device=args.device).eval()
    weights = parent_weights["projector.safetensors"]
    mapping = {"proj.0.weight": "linear_1.weight", "proj.2.weight": "linear_2.weight"}
    projector.load_state_dict({mapping.get(k, k): v for k, v in weights.items()}, strict=True)
    adapter = MoonViTV2Adapter.__new__(MoonViTV2Adapter)
    adapter.model_id = str(args.bundle)
    from transformers import AutoModel
    adapter.model = AutoModel.from_pretrained(args.bundle, trust_remote_code=True, dtype=dtype).to(args.device).eval()
    # Load the local Projector without resolving a Hub ID.
    adapter.projector = load_projector(dtype, str(args.bundle)).to(args.device)
    adapter.resolution, adapter.dtype, adapter.device, adapter.num_layers = 448, dtype, args.device, 27
    captured = {}
    handles = []
    for depth in adapter.depth_points():
        module = tower.encoder if depth == 27 else tower.encoder.blocks[depth - 1]
        handles.append(module.register_forward_hook(lambda m, a, y, depth=depth: captured.update({depth: y.detach()})))
    cases = []
    isolation = []
    for seed in range(3):
        pixels = np.random.default_rng(seed).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        inputs = collate([adapter.preprocess()(image)])
        with torch.inference_mode():
            if args.isolate:
                pixels_tensor = inputs["pixel_values"].to(args.device, dtype)
                grid = inputs["grid_thws"].to(args.device)
                actual_stem = adapter.model.patch_embed(pixels_tensor, grid)
                expected_stem = tower.patch_embed(pixels_tensor, grid)
                a_layer, r_layer = adapter.model.encoder.blocks[0], tower.encoder.blocks[0]
                actual_norm = a_layer.norm0(actual_stem)
                expected_norm = r_layer.norm0(actual_stem)
                actual_qkv = a_layer.wqkv(actual_norm)
                expected_qkv, _ = r_layer.wqkv(actual_norm)
                a_freq = adapter.model.encoder.rope_2d.get_freqs_cis(grid, device=pixels_tensor.device)
                r_freq = tower.encoder.rope_2d.get_freqs_cis(grid, device=pixels_tensor.device)
                packed = actual_qkv.reshape(-1, 3, 12, 128).clone()
                query, key, _ = packed.unbind(-3)
                apply_rope = a_layer.attention_qkvpacked.__func__.__globals__["apply_rope"]
                actual_query, actual_key = apply_rope(query, key, a_freq)
                if reference.HAS_TRITON:
                    reference.packed_qk_rope_(packed, a_freq)
                    expected_query, expected_key, _ = packed.unbind(-3)
                else:
                    expected_query = r_layer.apply_rotary_emb(query, a_freq.real.contiguous(), a_freq.imag.contiguous())
                    expected_key = r_layer.apply_rotary_emb(key, a_freq.real.contiguous(), a_freq.imag.contiguous())
                for name, actual_value, expected_value in (("stem", actual_stem, expected_stem),
                        ("first_norm", actual_norm, expected_norm), ("first_qkv", actual_qkv, expected_qkv),
                        ("rope_frequencies", torch.view_as_real(a_freq), torch.view_as_real(r_freq)),
                        ("rotated_query", actual_query, expected_query), ("rotated_key", actual_key, expected_key)):
                    delta_value = (actual_value.float()-expected_value.float()).abs()
                    isolation.append(dict(seed=seed, operation=name, exact=torch.equal(actual_value,expected_value),
                                          unequal_elements=int((actual_value!=expected_value).sum()), elements=actual_value.numel(),
                                          max_abs=float(delta_value.max()), relative_l2=float(delta_value.norm()/expected_value.float().norm())))
            merged = tower(inputs["pixel_values"].to(args.device, dtype), inputs["grid_thws"].to(args.device))
            expected = {("tower", depth): value[None] for depth, value in captured.items()}
            expected["merged", 27] = torch.stack(merged).flatten(2)
            expected["projected", 27] = projector(merged[0])[None]
            final = expected["tower", 27]
            raster_merge = final.reshape(1, 16, 2, 16, 2, 1024).permute(0, 1, 3, 2, 4, 5).reshape(1, 256, 4096)
            assert torch.equal(raster_merge, expected["merged", 27])
            for actual in adapter.extract(inputs, [str(seed)]):
                target = expected[actual.stage, actual.layer_index].cpu()
                assert torch.isfinite(actual.tokens).all() and torch.isfinite(target).all()
                delta = (actual.tokens.float() - target.float()).abs()
                atol, rtol = ((0.125, 0.03) if dtype == torch.bfloat16 else (0.003, 0.001))
                close = torch.isclose(actual.tokens, target, atol=atol, rtol=rtol)
                if not args.diagnostic:
                    torch.testing.assert_close(actual.tokens, target, atol=atol, rtol=rtol)
                actual_flat, target_flat = actual.tokens.float().flatten(), target.float().flatten()
                cases.append(dict(seed=seed, image_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), stage=actual.stage, layer=actual.layer_index, shape=list(target.shape), max_abs=float(delta.max()), mean_abs=float(delta.mean()), relative_l2=float((actual_flat-target_flat).norm()/target_flat.norm()), cosine=float(torch.nn.functional.cosine_similarity(actual_flat,target_flat,dim=0)), mismatched_elements=int((~close).sum()), elements=close.numel(), passed=bool(close.all()), atol=atol, rtol=rtol))
    for handle in handles:
        handle.remove()
    args.out.write_text(json.dumps(dict(checkpoint=args.checkpoint, source_revision=checkpoint.revision, reference_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), config_sha256=hashlib.sha256(args.parent_config.read_bytes()).hexdigest(), dtype=args.dtype, device=args.device, threads=args.threads, attention_substitution=args.attention, rotary_implementation=args.rotary, isolation=isolation, cases=cases, passed=all(c["passed"] for c in cases)), indent=2) + "\n")


if __name__ == "__main__":
    main()
