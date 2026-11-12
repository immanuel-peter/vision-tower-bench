"""Compare whole blocks, then each op under the adapter's own tensor shapes, against the reference."""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from huggingface_hub import hf_hub_download

from scripts import validate_mistral_large4 as v
from vtb.adapters import mistral_large4 as ml4

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev, dt = "cuda", torch.bfloat16
native = v.native_from_shard(Path(hf_hub_download(ml4.SOURCE_REPO, v.SHARD, revision=ml4.SOURCE_REVISION)))
params = ml4.source_json("params.json")
vision, layers = v.import_reference(v.fetch_reference(Path(sys.argv[1])))
layers.memory_efficient_attention = v.efficient_attention
ref = v.build_reference(vision, layers, params, native, dt, dev)
tower, projector = ml4.load_bundle_parts(ml4.RELEASE_DIR, dt)
tower = tower.to(dev)
pixels = ml4.Preprocess(448)(Image.open(sys.argv[2] + "/validation_006500.jpg")).to(dev, dt)


def st(a, b):
    a, b = a.reshape(b.shape), b
    return {"exact_fraction": (a == b).float().mean().item(), "bit_exact": bool(torch.equal(a, b)),
            "relative": ((a.double() - b.double()).norm() / b.double().norm()).item()}


out = {}
with torch.inference_mode():
    enc = ref.vision_encoder
    x2 = enc.ln_pre(enc.patch_conv(pixels[None]).squeeze(0).flatten(1).permute(1, 0))
    x3 = tower.ln_pre(tower.patch_conv(pixels[None]).flatten(2).transpose(1, 2))
    out["ln_pre_2d_vs_3d"] = st(x3[0], x2)
    out["strides"] = {"ref_2d": list(x2.stride()), "adapter_3d": list(x3.stride())}
    positions = torch.stack(torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij"), -1).reshape(-1, 2).to(dev)
    freqs = enc.freqs_cis[positions[:, 0], positions[:, 1]]
    table = ml4.rotary_table(32, 32, 128, 10000.0).to(dev)
    blk_r, blk_a = enc.transformer.layers[0], tower.transformer.layers[0]
    y_r = blk_r(x2, freqs_cis=freqs)
    y_a = ml4.block_forward(blk_a, x3, table)
    out["block0_whole"] = st(y_a[0], y_r)
    y_a_c = ml4.block_forward(blk_a, x3.contiguous(), table)
    out["block0_whole_contiguous_input"] = st(y_a_c[0], y_r)
    y_r_c = blk_r(x2.contiguous(), freqs_cis=freqs)
    out["ref_block0_contiguous_vs_strided"] = st(y_r_c, y_r)
    # op trace in adapter shapes, feeding the reference's intermediate values
    n_r = blk_r.attention_norm(x2)
    n_a = blk_a.attention_norm(x3)
    out["attention_norm_3d"] = st(n_a[0], n_r)
    att = blk_r.attention
    q_r = att.wq(n_r)
    q_a = blk_a.attention.q_proj(n_r[None])
    out["q_proj_3d_input_rows_permuted_back"] = st(
        q_a[0].view(-1, 16, 2, 64).transpose(2, 3).reshape(-1, 2048), q_r)
    q_a2 = blk_a.attention.q_proj(n_r)
    out["q_proj_2d_input"] = st(q_a2.view(-1, 16, 2, 64).transpose(2, 3).reshape(-1, 2048), q_r)
    xq, xk, xv = att.wq(n_r).view(-1, 16, 128), att.wk(n_r).view(-1, 16, 128), att.wv(n_r).view(-1, 16, 128)
    from mistral_inference.rope import apply_rotary_emb
    rq, rk = apply_rotary_emb(xq, xk, freqs_cis=freqs)
    o_ref = v.efficient_attention(rq[None], rk[None], xv[None], None)
    halves = lambda t: t.view(-1, 16, 64, 2).transpose(2, 3).reshape(-1, 16, 128)  # noqa: E731
    qa, ka, va = (halves(t)[None].transpose(1, 2) for t in (rq, rk, xv))
    va = xv[None].transpose(1, 2)
    with sdpa_kernel(ml4.ATTENTION_BACKENDS):
        o_a = torch.nn.functional.scaled_dot_product_attention(qa, ka, va, scale=128 ** -0.5)
    out["attention_halves_layout_vs_ref_layout"] = st(o_a.transpose(1, 2)[0], o_ref[0])
    with sdpa_kernel(ml4.ATTENTION_BACKENDS):
        o_i = torch.nn.functional.scaled_dot_product_attention(
            rq[None].transpose(1, 2), rk[None].transpose(1, 2), va, scale=128 ** -0.5)
    out["attention_interleaved_layout_vs_ref"] = st(o_i.transpose(1, 2)[0], o_ref[0])
    out["note"] = "attention halves vs interleaved reorders the q/k dot-product summation"
print(json.dumps(out, indent=1))
Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + "\n")
