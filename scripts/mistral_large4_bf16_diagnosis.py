#!/usr/bin/env python3
"""Localize BF16 differences between the converted Tower and the independent Mistral reference.

Part 1 (``--ops``) feeds identical BF16 inputs to one block of each
implementation and compares every operation: input norm, q/k/v projections,
rotary embedding, attention, output projection, MLP norm and MLP. Each
operation receives the reference's own input, so a difference cannot carry
over from an earlier operation.

Part 2 (``--forward``) runs whole forwards in BF16 and compares the converted
path with the reference directly, Stage by Stage, under variants that change
one numeric choice at a time:

- rope: ``hf`` rotates q and k in BF16 with BF16 cos/sin (HF Pixtral);
  ``fp32`` upcasts q and k, rotates with FP32 cos/sin and casts back, the
  operation order of mistral_inference.rope.apply_rotary_emb.
- attention: ``default`` lets SDPA pick its kernel, as HF does; ``efficient`` forces
  PyTorch's memory-efficient kernel, which PyTorch adapted from the xformers
  kernel the reference imports; ``explicit`` is float32 softmax attention.

The reference runs with ``efficient`` (closest to xformers) and ``explicit``.
The ``adapter`` variant is vtb.adapters.mistral_large4.tower_forward as shipped.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.pixtral import modeling_pixtral

from vtb import validate_mistral_large4 as validate
from vtb.adapters import mistral_large4 as ml4

DEPTHS = (4, 8, 11, 15, 19, 22, 26, 30)
HF_APPLY_ROTARY = modeling_pixtral.apply_rotary_pos_emb


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, e = actual.double().flatten(), expected.double().flatten()
    delta = a - e
    return {
        "relative": (delta.norm() / e.norm()).item() if e.norm() > 0 else float(delta.norm()),
        "cosine": torch.nn.functional.cosine_similarity(a, e, dim=0).item(),
        "max_abs": delta.abs().max().item(),
        "bit_exact_fraction": (actual == expected).float().mean().item(),
        "bit_exact": bool(torch.equal(actual, expected)),
    }


def rotary_fp32(q, k, cos, sin, unsqueeze_dim=1):
    """Mistral's order: rotate in float32, then cast back to the input dtype."""
    cos = cos.float().unsqueeze(unsqueeze_dim)
    sin = sin.float().unsqueeze(unsqueeze_dim)
    q32, k32 = q.float(), k.float()
    q_out = q32 * cos + modeling_pixtral.rotate_half(q32) * sin
    k_out = k32 * cos + modeling_pixtral.rotate_half(k32) * sin
    return q_out.to(q.dtype), k_out.to(k.dtype)


@contextlib.contextmanager
def converted_numerics(rope: str, attention: str, fp32_tables: bool):
    """Swap the converted model's rotary op and SDPA kernel for one forward."""
    original = modeling_pixtral.apply_rotary_pos_emb
    if rope == "fp32":
        modeling_pixtral.apply_rotary_pos_emb = rotary_fp32
    try:
        with attention_context(attention):
            yield
    finally:
        modeling_pixtral.apply_rotary_pos_emb = original


@contextlib.contextmanager
def attention_context(kind: str):
    if kind == "efficient":
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            yield
    elif kind == "flash":
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
    elif kind == "math":
        with sdpa_kernel(SDPBackend.MATH):
            yield
    else:
        yield


def reference_attention(kind: str):
    """xformers memory_efficient_attention substitutes for the reference."""
    return validate.REFERENCE_ATTENTION[kind]


def tower_forward(tower, pixels, rope_fp32_tables: bool):
    embeds = tower.patch_conv(pixels.to(tower.patch_conv.weight.dtype))
    rows, cols = embeds.shape[-2:]
    hidden = tower.ln_pre(embeds.flatten(2).transpose(1, 2))
    ys, xs = torch.meshgrid(torch.arange(rows, device=hidden.device), torch.arange(cols, device=hidden.device), indexing="ij")
    positions = torch.stack((ys.flatten(), xs.flatten()), dim=-1)
    table_input = hidden.float() if rope_fp32_tables else hidden
    cos, sin = tower.patch_positional_embedding(table_input, positions)
    captured = {}
    for depth, layer in enumerate(tower.transformer.layers, start=1):
        hidden = layer(hidden, attention_mask=None, position_embeddings=(cos, sin))
        if depth in DEPTHS:
            captured[depth] = hidden
    return hidden, captured


def converted_stages(tower, projector, pixels, rope: str, attention: str) -> dict:
    with converted_numerics(rope, attention, rope == "fp32"):
        final, captured = tower_forward(tower, pixels, rope == "fp32")
    merged, projected = ml4.projector_stages(projector, final, pixels.shape[-1] // ml4.PATCH_SIZE)
    stages = {f"tower_L{d}": captured[d][0].cpu() for d in DEPTHS}
    stages["merged"] = merged[0].cpu()
    stages["projected"] = projected[0].cpu()
    return stages


def block_ops(reference, tower, pixels: torch.Tensor, layers_module) -> list[dict]:
    """Op-by-op comparison inside block 1, with each op fed the reference's input."""
    from mistral_inference.rope import apply_rotary_emb

    rows = []
    encoder = reference.vision_encoder
    embeds = encoder.patch_conv(pixels[None]).squeeze(0)
    src = encoder.ln_pre(embeds.flatten(1).permute(1, 0))
    conv_embeds = tower.patch_conv(pixels[None])
    conv = tower.ln_pre(conv_embeds.flatten(2).transpose(1, 2))[0]
    rows.append({"op": "patch_conv", **stats(conv_embeds[0], embeds)})
    rows.append({"op": "ln_pre", **stats(conv, src)})

    side = pixels.shape[-1] // 14
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    positions = torch.stack((ys.flatten(), xs.flatten()), dim=-1).to(src.device)
    freqs = encoder.freqs_cis[positions[:, 0], positions[:, 1]]
    cos, sin = tower.patch_positional_embedding(src[None], positions)

    block_src = encoder.transformer.layers[0]
    block_hf = tower.transformer.layers[0]
    x = src
    n_src = block_src.attention_norm(x)
    n_hf = block_hf.attention_norm(x[None])[0]
    rows.append({"op": "attention_norm", **stats(n_hf, n_src)})

    heads, dim = 16, 128
    q_src = block_src.attention.wq(n_src).view(-1, heads, dim)
    k_src = block_src.attention.wk(n_src).view(-1, heads, dim)
    v_src = block_src.attention.wv(n_src).view(-1, heads, dim)
    q_hf = block_hf.attention.q_proj(n_src)
    k_hf = block_hf.attention.k_proj(n_src)
    v_hf = block_hf.attention.v_proj(n_src)

    def to_interleaved(t):  # (T, 2048) halves layout -> (T, H, D) interleaved
        return t.view(-1, heads, 2, dim // 2).transpose(2, 3).reshape(-1, heads, dim)

    rows.append({"op": "q_proj (rows permuted back)", **stats(to_interleaved(q_hf), q_src)})
    rows.append({"op": "k_proj (rows permuted back)", **stats(to_interleaved(k_hf), k_src)})
    rows.append({"op": "v_proj", **stats(v_hf.view(-1, heads, dim), v_src)})

    rq_src, rk_src = apply_rotary_emb(q_src, k_src, freqs_cis=freqs)
    halves = lambda t: t.view(-1, heads, dim // 2, 2).transpose(2, 3).reshape(-1, heads, dim)  # noqa: E731
    q_in, k_in = halves(q_src).transpose(0, 1)[None], halves(k_src).transpose(0, 1)[None]
    rq_hf, rk_hf = HF_APPLY_ROTARY(q_in, k_in, cos, sin, unsqueeze_dim=0)
    rq_32, rk_32 = rotary_fp32(q_in, k_in, cos.float(), sin.float(), unsqueeze_dim=0)
    cos32, sin32 = tower.patch_positional_embedding(src[None].float(), positions)
    rq_32t, rk_32t = rotary_fp32(q_in, k_in, cos32, sin32, unsqueeze_dim=0)
    back = lambda t: to_interleaved(t[0].transpose(0, 1).reshape(-1, heads * dim))  # noqa: E731
    rows.append({"op": "rope q, HF (BF16 tables, BF16 math)", **stats(back(rq_hf), rq_src)})
    rows.append({"op": "rope k, HF (BF16 tables, BF16 math)", **stats(back(rk_hf), rk_src)})
    rows.append({"op": "rope q, FP32 math on BF16 tables", **stats(back(rq_32), rq_src)})
    rows.append({"op": "rope q, FP32 math on FP32 tables", **stats(back(rq_32t), rq_src)})
    rows.append({"op": "rope k, FP32 math on FP32 tables", **stats(back(rk_32t), rk_src)})
    table = ml4.rotary_table(side, side, dim, 10000.0).to(src.device)
    rows.append({"op": "rope q, adapter (source table, complex multiply)", **stats(ml4.rotate(q_in, table)[0].transpose(0, 1), rq_src)})
    rows.append({"op": "rope k, adapter (source table, complex multiply)", **stats(ml4.rotate(k_in, table)[0].transpose(0, 1), rk_src)})

    # Attention on identical rotated inputs, in each layout.
    q4, k4, v4 = rq_src[None], rk_src[None], v_src[None]
    att_explicit = validate.memory_efficient_attention(q4, k4, v4)[0]
    for kind in ("efficient", "flash", "math", "default"):
        try:
            with attention_context(kind):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q4.transpose(1, 2), k4.transpose(1, 2), v4.transpose(1, 2), scale=dim**-0.5
                ).transpose(1, 2)[0]
            rows.append({"op": f"attention sdpa[{kind}] vs explicit fp32", **stats(out, att_explicit)})
        except RuntimeError as error:
            rows.append({"op": f"attention sdpa[{kind}]", "error": str(error)[:200]})

    attn = att_explicit.reshape(-1, heads * dim)
    o_src = block_src.attention.wo(attn)
    o_hf = block_hf.attention.o_proj(attn)
    rows.append({"op": "o_proj", **stats(o_hf, o_src)})
    h = x + o_src
    f_src = block_src.ffn_norm(h)
    f_hf = block_hf.ffn_norm(h[None])[0]
    rows.append({"op": "ffn_norm", **stats(f_hf, f_src)})
    m_src = block_src.feed_forward(f_src)
    m_hf = block_hf.feed_forward(f_src)
    rows.append({"op": "feed_forward (SwiGLU)", **stats(m_hf, m_src)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=ml4.RELEASE_DIR)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--resolutions", nargs="+", type=int, default=[448])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--forward", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()

    from huggingface_hub import hf_hub_download

    native = validate.native_from_shard(
        Path(hf_hub_download(ml4.SOURCE_REPO, validate.SHARD, revision=ml4.SOURCE_REVISION))
    )
    params = ml4.source_json("params.json")
    vision, layers = validate.import_reference(validate.fetch_reference(args.reference_root))
    images = validate.synthetic_images(args.out.parent / "diagnosis-images") + [Path(p) for p in args.images]
    report = {
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "torch": torch.__version__,
        "images": [{"path": str(p), "sha256": validate.sha256_file(p)} for p in images],
        "hf_default_attention": None,
    }
    dtype = torch.bfloat16
    tower, projector = ml4.load_bundle_parts(args.bundle, dtype)
    tower, projector = tower.to(args.device), projector.to(args.device)
    report["hf_default_attention"] = tower.config._attn_implementation

    if args.ops:
        reference = validate.build_reference(vision, layers, params, native, dtype, args.device)
        report["ops"] = {}
        for resolution in args.resolutions:
            pixels = ml4.Preprocess(resolution)(Image.open(images[-1])).to(args.device, dtype)
            with torch.inference_mode():
                report["ops"][str(resolution)] = block_ops(reference, tower, pixels, layers)
        del reference

    if args.forward:
        report["forward"] = []
        reference_runs = {}
        for kind in ("efficient", "explicit"):
            layers.memory_efficient_attention = reference_attention(kind)
            reference = validate.build_reference(vision, layers, params, native, dtype, args.device)
            for resolution in args.resolutions:
                preprocess = ml4.Preprocess(resolution)
                for path in images:
                    pixels = preprocess(Image.open(path)).to(args.device, dtype)
                    with torch.inference_mode():
                        reference_runs[(kind, resolution, path.name)] = {
                            k: v.cpu() for k, v in reference(pixels, DEPTHS).items()
                        }
            del reference
            torch.cuda.empty_cache()
        layers.memory_efficient_attention = validate.memory_efficient_attention
        variants = [(r, a) for r in ("hf", "fp32") for a in ("default", "efficient", "math")] + [("adapter", "adapter")]
        for resolution in args.resolutions:
            preprocess = ml4.Preprocess(resolution)
            for path in images:
                pixels = preprocess(Image.open(path))[None].to(args.device, dtype)
                for rope, attention in variants:
                    with torch.inference_mode():
                        if rope == "adapter":
                            final, captured = ml4.tower_forward(tower, pixels, DEPTHS)
                            merged, projected = ml4.projector_stages(projector, final, pixels.shape[-1] // ml4.PATCH_SIZE)
                            stages = {f"tower_L{d}": captured[d][0].cpu() for d in DEPTHS}
                            stages |= {"merged": merged[0].cpu(), "projected": projected[0].cpu()}
                        else:
                            stages = converted_stages(tower, projector, pixels, rope, attention)
                    for kind in ("efficient", "explicit"):
                        ref = reference_runs[(kind, resolution, path.name)]
                        report["forward"].append({
                            "resolution": resolution, "image": path.name,
                            "converted": {"rope": rope, "attention": attention},
                            "reference_attention": kind,
                            "stages": {stage: stats(stages[stage], ref[stage]) for stage in ref},
                        })
    report["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
