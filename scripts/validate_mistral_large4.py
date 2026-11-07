"""Check the PixtralViT-Large-4 Tower and Projector against an independent Mistral forward.

The reference is mistral-inference 1.6.0, loaded from its SHA256-pinned wheel.
It reads the parent's native tensor names (``wq``, ``w1``, ``patch_merger``,
``vision_language_adapter``) and applies RoPE as complex multiplication on
interleaved channel pairs, so it never touches the HF conversion. The
reference's vision path is the four lines of
``Transformer.embed_vision_language_features``: ``vision_encoder`` ->
``pre_mm_projector_norm`` -> ``patch_merger`` -> ``vision_language_adapter``.

mistral-inference imports xformers for ``memory_efficient_attention``. This
script registers a substitute module that computes the same softmax(q k^T /
sqrt(d)) v with float32 accumulation, head by head, in the reference's own
(batch, sequence, head, dim) layout. Its ``BlockDiagonalMask`` keeps the
xformers semantics. ``tests/test_mistral_large4.py`` checks both against
direct formulas. Only one image is packed per reference call, and the
reference ``TransformerBlock`` does not forward the mask to attention anyway.
"""

import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
import types
import zipfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import requests
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from safetensors import safe_open

from vtb.adapters import mistral_large4 as ml4

REFERENCE_PACKAGE = "mistral-inference"
REFERENCE_VERSION = "1.6.0"
REFERENCE_WHEEL = "mistral_inference-1.6.0-py3-none-any.whl"
REFERENCE_SHA256 = "ab55f27381836d81e67b323eea21bc0f5f483a9eff8c603d04d9fabde925e836"
SHARD = "consolidated-00053-of-00054.safetensors"
# Decided before the first comparison. FP32 runs disable TF32, so any layout
# mistake shows up far above this bound.
FP32_RELATIVE_TOLERANCE = 1e-4
# A BF16 candidate may drift from the FP32 reference by at most 1.5 times the
# reference's own BF16 drift plus 0.005. The first run also required cosine
# 0.999. Tower L30 carries activations up to 2.7e5 in three channels, so the
# reference's own BF16 forward reaches only cosine 0.984 there. The cosine
# bound is now relative to that floor; the fixed bound is still reported.
BF16_FLOOR_FACTOR = 1.5
BF16_FLOOR_SLACK = 0.005
BF16_MIN_COSINE = 0.999
BF16_COSINE_SLACK = 0.002
# Negative controls must miss by at least this relative error.
CONTROL_MIN_RELATIVE = 0.05


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- reference


def fetch_reference(root: Path) -> Path:
    """Download and unpack the pinned wheel; return the directory to import from."""
    unpacked = root / f"mistral_inference_{REFERENCE_VERSION}"
    wheel = root / REFERENCE_WHEEL
    if not wheel.is_file():
        root.mkdir(parents=True, exist_ok=True)
        release = requests.get(f"https://pypi.org/pypi/{REFERENCE_PACKAGE}/{REFERENCE_VERSION}/json", timeout=60)
        release.raise_for_status()
        url = next(item["url"] for item in release.json()["urls"] if item["filename"] == REFERENCE_WHEEL)
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        wheel.write_bytes(response.content)
    if sha256_file(wheel) != REFERENCE_SHA256:
        raise ValueError(f"{wheel} does not match the pinned SHA256")
    if not (unpacked / "mistral_inference" / "vision_encoder.py").is_file():
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(unpacked)
    return unpacked


class BlockDiagonalMask:
    """xformers BlockDiagonalMask: tokens attend only within their own block."""

    def __init__(self, seqlens: list[int]):
        self.seqlens = list(seqlens)

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        if kv_seqlen is not None and list(kv_seqlen) != list(q_seqlen):
            raise NotImplementedError("only self-attention blocks are supported")
        return cls(q_seqlen)

    def materialize(self, shape, dtype=torch.float32, device="cpu") -> torch.Tensor:
        total = sum(self.seqlens)
        mask = torch.full((total, total), float("-inf"), dtype=dtype, device=device)
        start = 0
        for length in self.seqlens:
            mask[start:start + length, start:start + length] = 0
            start += length
        return mask.expand(shape)


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None):
    """softmax(q k^T * scale + bias) v for (batch, seq, heads, dim) inputs, float32 accumulation."""
    if p:
        raise NotImplementedError("dropout is not used by the reference")
    batch, length, heads, dim = query.shape
    scale = dim**-0.5 if scale is None else scale
    bias = None
    if attn_bias is not None:
        bias = attn_bias.materialize((length, key.shape[1]), dtype=torch.float32, device=query.device)
    out = torch.empty_like(query)
    for b in range(batch):
        for h in range(heads):
            q = query[b, :, h].float()
            k = key[b, :, h].float()
            v = value[b, :, h].float()
            scores = (q @ k.T) * scale
            if bias is not None:
                scores = scores + bias
            out[b, :, h] = (torch.softmax(scores, dim=-1) @ v).to(query.dtype)
    return out


class _Placeholder:
    pass


def install_reference_substitutes() -> None:
    """Register the xformers and simple_parsing names mistral_inference imports."""
    attn_bias = types.ModuleType("xformers.ops.fmha.attn_bias")
    attn_bias.BlockDiagonalMask = BlockDiagonalMask
    for name in ("AttentionBias", "BlockDiagonalCausalMask", "BlockDiagonalCausalWithOffsetPaddedKeysMask"):
        setattr(attn_bias, name, type(name, (_Placeholder,), {}))
    fmha = types.ModuleType("xformers.ops.fmha")
    fmha.memory_efficient_attention = memory_efficient_attention
    fmha.attn_bias = attn_bias
    ops = types.ModuleType("xformers.ops")
    ops.fmha = fmha
    xformers = types.ModuleType("xformers")
    xformers.ops = ops
    helpers = types.ModuleType("simple_parsing.helpers")
    helpers.Serializable = type("Serializable", (), {})
    simple_parsing = types.ModuleType("simple_parsing")
    simple_parsing.helpers = helpers
    sys.modules.update({
        "xformers": xformers,
        "xformers.ops": ops,
        "xformers.ops.fmha": fmha,
        "xformers.ops.fmha.attn_bias": attn_bias,
        "simple_parsing": simple_parsing,
        "simple_parsing.helpers": helpers,
    })


def import_reference(unpacked: Path):
    install_reference_substitutes()
    if str(unpacked) not in sys.path:
        sys.path.insert(0, str(unpacked))
    vision = importlib.import_module("mistral_inference.vision_encoder")
    layers = importlib.import_module("mistral_inference.transformer_layers")
    if not Path(vision.__file__).resolve().is_relative_to(unpacked.resolve()):
        raise ImportError(f"mistral_inference resolved outside the pinned wheel: {vision.__file__}")
    return vision, layers


class Reference(torch.nn.Module):
    """The parent's vision path, built from mistral-inference classes and native tensor names."""

    def __init__(self, vision, layers, params: dict):
        super().__init__()
        encoder = params["vision_encoder"]
        names = {item.name for item in fields(vision.VisionEncoderArgs)}
        args = vision.VisionEncoderArgs(**{key: value for key, value in encoder.items() if key in names})
        self.args = args
        self.vision_encoder = vision.VisionTransformer(args)
        # transformer.py, Transformer.__init__: the same constructors and eps.
        self.vision_language_adapter = vision.VisionLanguageAdapter(args.hidden_size, params["dim"], args.adapter_bias)
        if not args.add_pre_mm_projector_layer_norm or args.mm_projector_id != "patch_merge":
            raise ValueError("expected the pre-projector norm and the patch merger")
        self.pre_mm_projector_norm = layers.RMSNorm(args.hidden_size, eps=1e-5)
        self.patch_merger = vision.PatchMerger(args.hidden_size, args.spatial_merge_size)

    def forward(self, image: torch.Tensor, depths: tuple[int, ...]) -> dict[str, torch.Tensor]:
        captured = {}
        handles = [
            self.vision_encoder.transformer.layers[depth - 1].register_forward_hook(
                lambda _m, _a, out, depth=depth: captured.__setitem__(depth, out)
            )
            for depth in depths
        ]
        try:
            # transformer.py, Transformer.embed_vision_language_features.
            features = self.vision_encoder([image])
            features = self.pre_mm_projector_norm(features)
            patch = self.args.patch_size
            sizes = [(image.shape[1] // patch, image.shape[2] // patch)]
            merged = self.patch_merger(features, image_sizes=sizes)
            projected = self.vision_language_adapter(merged)
        finally:
            for handle in handles:
                handle.remove()
        stages = {f"tower_L{depth}": captured[depth] for depth in depths}
        stages["merged"] = merged
        stages["projected"] = projected
        return stages


def build_reference(vision, layers, params: dict, native: dict[str, torch.Tensor], dtype, device) -> Reference:
    reference = Reference(vision, layers, params)
    state = {}
    for name, tensor in native.items():
        state[name] = tensor
    missing = reference.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys
    return reference.to(dtype=dtype, device=device).eval()


# ---------------------------------------------------------------- candidate


def candidate_stages(tower, projector, pixels: torch.Tensor, depths: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """The bench adapter's path: batched Tower forward, then the HF Projector."""
    final, captured = ml4.tower_forward(tower, pixels, depths)
    merged, projected = ml4.projector_stages(projector, final, pixels.shape[-1] // ml4.PATCH_SIZE)
    stages = {f"tower_L{depth}": captured[depth][0] for depth in depths}
    stages["merged"] = merged[0]
    stages["projected"] = projected[0]
    return stages


def hf_packed_stages(tower, projector, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    """HF's own packed PixtralVisionModel.forward and Mistral3MultiModalProjector.forward."""
    sizes = [tuple(pixels.shape[-2:])] * len(pixels)
    final = tower(pixels, image_sizes=sizes).last_hidden_state.squeeze(0)
    projected = projector(final, sizes)
    return {f"tower_L{tower.config.num_hidden_layers}": final, "projected": projected}


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, e = actual.double().flatten(), expected.double().flatten()
    if actual.shape != expected.shape:
        raise ValueError(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    delta = a - e
    return {
        "shape": list(actual.shape),
        "max_abs": delta.abs().max().item(),
        "reference_max_abs": e.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "relative": (delta.norm() / e.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, e, dim=0).item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


# ---------------------------------------------------------------- controls


def control_parts(native: dict[str, torch.Tensor], source: dict, dtype, device, name: str):
    """Build deliberately wrong candidates; each must miss the reference."""
    tower_state, projector_state = ml4.convert(native)
    if name == "plain_rename_without_rope_permutation":
        tower_state = {
            key: ml4.halves_to_interleaved(value) if ml4.is_rotary("vision_tower." + key) else value
            for key, value in tower_state.items()
        }
    elif name == "merger_kernel_major_input":
        weight = projector_state["patch_merger.merging_layer.weight"]
        out, width = weight.shape[0], weight.shape[1] // 4
        # Read the 8192 inputs as kernel-major (k * D + c) instead of channel-major (c * 4 + k).
        projector_state = dict(projector_state)
        projector_state["patch_merger.merging_layer.weight"] = (
            weight.view(out, 4, width).transpose(1, 2).reshape(out, 4 * width).contiguous()
        )
    elif name not in {"final_norm_on_tower", "norm_eps_1e-6"}:
        raise KeyError(name)
    tower, projector = ml4.build_parts(
        tower_state, projector_state, ml4.vision_config(source), ml4.projector_config(source), dtype
    )
    if name == "norm_eps_1e-6":
        for module in [*tower.modules(), *projector.modules()]:
            if hasattr(module, "variance_epsilon"):
                module.variance_epsilon = 1e-6
    return tower.to(device), projector.to(device)


def final_rms_norm(tokens: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    flat = tokens.float()
    return (flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)).to(tokens.dtype)


# ---------------------------------------------------------------- inputs


def synthetic_images(out: Path) -> list[Path]:
    """Two fixed images: seeded noise and a smooth pattern with edges."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    noise = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:600, 0:800].astype(np.float32)
    pattern = np.stack([
        127 + 120 * np.sin(xs / 37.0),
        127 + 120 * np.cos(ys / 23.0),
        ((xs // 80 + ys // 80) % 2) * 200 + 25,
    ], axis=-1).clip(0, 255).astype(np.uint8)
    paths = []
    for name, array in (("seeded_noise.png", noise), ("pattern.png", pattern)):
        path = out / name
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


def native_from_shard(shard: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(shard), framework="pt") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys() if name.startswith(ml4.SOURCE_PREFIXES)}


def identity_report(shard: Path, bundle: Path | None) -> tuple[dict, dict[str, torch.Tensor]]:
    started = time.time()
    metadata = HfApi().model_info(ml4.SOURCE_REPO, revision=ml4.SOURCE_REVISION, files_metadata=True)
    lfs = next(item.lfs.sha256 for item in metadata.siblings if item.rfilename == SHARD)
    digest = sha256_file(shard)
    if digest != lfs:
        raise ValueError(f"{shard} SHA256 {digest} does not match the pinned LFS {lfs}")
    native = native_from_shard(shard)
    index = ml4.source_index()
    if set(native) != set(index) or set(index.values()) != {SHARD}:
        raise ValueError("local shard tensors differ from the pinned index")
    ranged = ml4.source_weights()
    report = {
        "source_repo": ml4.SOURCE_REPO,
        "source_revision": ml4.SOURCE_REVISION,
        "resolved_commit": metadata.sha,
        "shard": SHARD,
        "shard_sha256": digest,
        "shard_lfs_sha256": lfs,
        "vision_tensors": len(native),
        "tower_tensors": sum(name.startswith(ml4.TOWER_PREFIX) for name in native),
        "projector_tensors": sum(not name.startswith(ml4.TOWER_PREFIX) for name in native),
        "parameters": sum(t.numel() for t in native.values()),
        "tower_parameters": sum(t.numel() for n, t in native.items() if n.startswith(ml4.TOWER_PREFIX)),
        "projector_parameters": sum(t.numel() for n, t in native.items() if not n.startswith(ml4.TOWER_PREFIX)),
        "dtypes": sorted({str(t.dtype) for t in native.values()}),
        "range_fetch_bit_exact": all(torch.equal(ranged[name], tensor) for name, tensor in native.items()),
        "range_fetch_names_equal": set(ranged) == set(native),
    }
    tower_state, projector_state = ml4.convert(native)
    if bundle is not None and (bundle / "model.safetensors").is_file():
        from safetensors.torch import load_file

        released = {"vision_tower." + k: v for k, v in load_file(bundle / "model.safetensors").items()}
        released.update({
            "multi_modal_projector." + k: v for k, v in load_file(bundle / "projector.safetensors").items()
        })
        manifest = json.loads((bundle / "export_manifest.json").read_text())
        drift = {}
        exact = True
        for name, tensor in native.items():
            hf_name = ml4.remap_key(name)
            restored = released[hf_name]
            if ml4.is_rotary(hf_name):
                restored = ml4.halves_to_interleaved(restored)
            same = torch.equal(restored, tensor)
            exact &= same
            a, b = restored.double().flatten(), tensor.double().flatten()
            drift[name] = {
                "release_name": hf_name,
                "bit_exact": same,
                "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                "relative_l2": ((a - b).norm() / b.norm()).item(),
                "sha256_matches_manifest": manifest["tensor_sha256"][name] == tensor_sha256(tensor),
            }
        report["release"] = {
            "bundle": str(bundle),
            "tensors": len(released),
            "names_cover_source_once": sorted(released) == sorted(ml4.remap_key(n) for n in native),
            "bit_exact_after_inverse_permutation": exact,
            "manifest_sha256_all_match": all(item["sha256_matches_manifest"] for item in drift.values()),
            "max_relative_l2": max(item["relative_l2"] for item in drift.values()),
            "min_cosine": min(item["cosine"] for item in drift.values()),
            "per_tensor": drift,
        }
    rotary = [name for name in native if ml4.is_rotary(ml4.remap_key(name))]
    report["rope_permutation"] = {
        "tensors": len(rotary),
        "round_trip_bit_exact": all(
            torch.equal(ml4.halves_to_interleaved(ml4.interleaved_to_halves(native[name])), native[name])
            for name in rotary
        ),
        "min_relative_l2_permuted_vs_native": min(
            ((ml4.interleaved_to_halves(native[n]).double() - native[n].double()).norm() / native[n].double().norm()).item()
            for n in rotary
        ),
    }
    report["converted_tensors"] = {"tower": len(tower_state), "projector": len(projector_state)}
    report["seconds"] = round(time.time() - started, 1)
    return report, native


# ---------------------------------------------------------------- main


def run_parity(native, args, report) -> None:
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = ml4.source_json("config.json")
    params = ml4.source_json("params.json")
    vision, layers = import_reference(fetch_reference(args.reference_root))
    report["reference"] = {
        "package": REFERENCE_PACKAGE,
        "version": REFERENCE_VERSION,
        "wheel_sha256": REFERENCE_SHA256,
        "vision_encoder_py_sha256": sha256_file(Path(vision.__file__)),
        "transformer_layers_py_sha256": sha256_file(Path(layers.__file__)),
        "attention_substitute": "float32 softmax(q k^T / sqrt(d)) v per head in (batch, seq, head, dim) layout",
        "params_vision_encoder": params["vision_encoder"],
    }

    images = synthetic_images(args.out.parent / "parity-images") + [Path(p) for p in args.images]
    report["images"] = [{"path": str(p), "sha256": sha256_file(p)} for p in images]
    depths = tuple(round(30 * (k + 1) / 8) for k in range(8))
    report["depths"] = list(depths)
    if args.bundle is not None and (args.bundle / "model.safetensors").is_file():
        candidate_source = f"release bundle {args.bundle}"
        load_candidate = lambda dtype: ml4.load_bundle_parts(args.bundle, dtype)  # noqa: E731
    else:
        candidate_source = "convert() of the SHA256-verified shard tensors"
        load_candidate = lambda dtype: ml4.build_parts(  # noqa: E731
            *ml4.convert(native), ml4.vision_config(source), ml4.projector_config(source), dtype
        )
    report["candidate"] = candidate_source
    report["eps"] = {}
    report["cases"] = []
    report["controls"] = []

    for resolution in args.resolutions:
        preprocess = ml4.Preprocess(resolution)
        pixels = [preprocess(Image.open(p)) for p in images]
        outputs: dict[str, dict] = {}
        for dtype in (torch.float32, torch.bfloat16):
            label = str(dtype).removeprefix("torch.")
            reference = build_reference(vision, layers, params, native, dtype, device)
            if not report["eps"]:
                report["eps"]["reference"] = {
                    "ln_pre": reference.vision_encoder.ln_pre.eps,
                    "block_norms": sorted({
                        m.eps for m in reference.vision_encoder.transformer.modules() if isinstance(m, layers.RMSNorm)
                    }),
                    "pre_mm_projector_norm": reference.pre_mm_projector_norm.eps,
                    "final_norm_modules_after_last_block": [
                        name for name, _ in reference.vision_encoder.named_children()
                        if name not in {"patch_conv", "ln_pre", "transformer"}
                    ],
                }
            with torch.inference_mode():
                outputs[f"reference_{label}"] = [
                    {k: v.cpu() for k, v in reference(x.to(device, dtype), depths).items()} for x in pixels
                ]
            del reference
            torch.cuda.empty_cache()
            tower, projector = load_candidate(dtype)
            tower, projector = tower.to(device), projector.to(device)
            if "candidate" not in report["eps"]:
                report["eps"]["candidate"] = {
                    "tower": sorted({m.variance_epsilon for m in tower.modules() if hasattr(m, "variance_epsilon")}),
                    "projector_norm": projector.norm.variance_epsilon,
                    "rope_inv_freq_dtype": str(tower.patch_positional_embedding.inv_freq.dtype),
                }
            with torch.inference_mode():
                outputs[f"adapter_{label}"] = [
                    {k: v.cpu() for k, v in candidate_stages(tower, projector, x[None].to(device, dtype), depths).items()}
                    for x in pixels
                ]
                outputs[f"hf_packed_{label}"] = [
                    {k: v.cpu() for k, v in hf_packed_stages(tower, projector, x[None].to(device, dtype)).items()}
                    for x in pixels
                ]
                batch = torch.stack(pixels).to(device, dtype)
                batched = candidate_stages_batch(tower, projector, batch, depths)
            outputs[f"adapter_batch_{label}"] = batched
            del tower, projector
            torch.cuda.empty_cache()

        for index, path in enumerate(images):
            ref32 = outputs["reference_float32"][index]
            ref16 = outputs["reference_bfloat16"][index]
            for stage in ref32:
                case = {
                    "resolution": resolution,
                    "image": path.name,
                    "stage": stage,
                    "adapter_fp32": compare(outputs["adapter_float32"][index][stage], ref32[stage]),
                    "adapter_bf16": compare(outputs["adapter_bfloat16"][index][stage], ref32[stage]),
                    "reference_bf16_floor": compare(ref16[stage], ref32[stage]),
                    "adapter_bf16_vs_reference_bf16": compare(outputs["adapter_bfloat16"][index][stage], ref16[stage]),
                    "batched_equals_single_bf16": torch.equal(
                        outputs["adapter_batch_bfloat16"][stage][index], outputs["adapter_bfloat16"][index][stage]
                    ),
                    "batched_vs_single_fp32": compare(
                        outputs["adapter_batch_float32"][stage][index], outputs["adapter_float32"][index][stage]
                    ),
                }
                if stage in outputs["hf_packed_float32"][index]:
                    case["hf_packed_fp32"] = compare(outputs["hf_packed_float32"][index][stage], ref32[stage])
                    case["hf_packed_bf16"] = compare(outputs["hf_packed_bfloat16"][index][stage], ref32[stage])
                case["pass"] = case_passes(case)
                report["cases"].append(case)

        controls = [
            "plain_rename_without_rope_permutation",
            "merger_kernel_major_input",
            "final_norm_on_tower",
            "norm_eps_1e-6",
        ]
        for name in controls:
            tower, projector = control_parts(native, source, torch.float32, device, name)
            with torch.inference_mode():
                stages = candidate_stages(tower, projector, pixels[0][None].to(device), depths)
            if name == "final_norm_on_tower":
                stages = {f"tower_L{depths[-1]}": final_rms_norm(stages[f"tower_L{depths[-1]}"])}
            del tower, projector
            torch.cuda.empty_cache()
            ref32 = outputs["reference_float32"][0]
            measured = {stage: compare(value.cpu(), ref32[stage]) for stage, value in stages.items()}
            worst = max(item["relative"] for item in measured.values())
            report["controls"].append({
                "resolution": resolution,
                "image": images[0].name,
                "control": name,
                "worst_stage": max(measured, key=lambda s: measured[s]["relative"]),
                "worst_relative": worst,
                "stages": measured,
                # The eps control is a sensitivity measurement, not a required failure.
                "detected": worst >= CONTROL_MIN_RELATIVE if name != "norm_eps_1e-6" else None,
            })


def candidate_stages_batch(tower, projector, pixels, depths) -> dict[str, torch.Tensor]:
    final, captured = ml4.tower_forward(tower, pixels, depths)
    merged, projected = ml4.projector_stages(projector, final, pixels.shape[-1] // ml4.PATCH_SIZE)
    stages = {f"tower_L{depth}": captured[depth].cpu() for depth in depths}
    stages["merged"] = merged.cpu()
    stages["projected"] = projected.cpu()
    return stages


def case_passes(case: dict) -> bool:
    fp32_ok = case["adapter_fp32"]["relative"] <= FP32_RELATIVE_TOLERANCE
    floor = case["reference_bf16_floor"]["relative"]
    bf16 = case["adapter_bf16"]
    floor_cosine = case["reference_bf16_floor"]["cosine"]
    bf16_ok = (
        bf16["relative"] <= BF16_FLOOR_FACTOR * floor + BF16_FLOOR_SLACK
        and bf16["cosine"] >= floor_cosine - BF16_COSINE_SLACK
    )
    case["fixed_bf16_cosine_0.999"] = bf16["cosine"] >= BF16_MIN_COSINE
    finite = all(item["finite"] for item in case.values() if isinstance(item, dict))
    packed_ok = "hf_packed_fp32" not in case or case["hf_packed_fp32"]["relative"] <= FP32_RELATIVE_TOLERANCE
    return bool(fp32_ok and bf16_ok and finite and packed_ok)


def validate(
    out: Path,
    reference_root: Path,
    bundle: Path | None = None,
    images: list[str] | None = None,
    resolutions: list[int] | None = None,
    device: str = "cuda",
    identity_only: bool = False,
) -> dict:
    args = argparse.Namespace(
        out=out, reference_root=reference_root, bundle=bundle, images=images or [],
        resolutions=resolutions or [448, 1540], device=device,
    )
    started = time.time()
    shard = Path(hf_hub_download(ml4.SOURCE_REPO, SHARD, revision=ml4.SOURCE_REVISION))
    identity, native = identity_report(shard, bundle)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
        },
        "tolerances": {
            "fp32_relative": FP32_RELATIVE_TOLERANCE,
            "bf16_floor_factor": BF16_FLOOR_FACTOR,
            "bf16_floor_slack": BF16_FLOOR_SLACK,
            "bf16_cosine_slack_below_reference_floor": BF16_COSINE_SLACK,
            "bf16_fixed_cosine_reported_not_gating": BF16_MIN_COSINE,
            "control_min_relative": CONTROL_MIN_RELATIVE,
            "tf32": False,
        },
        "identity": identity,
    }
    if not identity_only:
        run_parity(native, args, report)
        report["all_cases_pass"] = all(case["pass"] for case in report["cases"])
        report["all_controls_detected"] = all(c["detected"] for c in report["controls"] if c["detected"] is not None)
    report["seconds"] = round(time.time() - started, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True, help="where to keep the pinned wheel")
    parser.add_argument("--bundle", type=Path, default=None, help="release directory to check instead of convert()")
    parser.add_argument("--images", nargs="*", default=[], help="real images added to the two synthetic ones")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[448, 1540])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.out, args.reference_root, args.bundle, args.images, args.resolutions, args.device, args.identity_only
    )
    summary = {key: report.get(key) for key in ("all_cases_pass", "all_controls_detected", "seconds")}
    print(json.dumps(summary, indent=2))
    if not args.identity_only and not (report["all_cases_pass"] and report["all_controls_detected"]):
        raise SystemExit("parity failed")


if __name__ == "__main__":
    main()
