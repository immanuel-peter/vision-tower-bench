"""Chain all 30 blocks of both implementations from one input; locate the first non-identical layer and op."""
import json, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from huggingface_hub import hf_hub_download
from scripts import validate_mistral_large4 as v
from vtb.adapters import mistral_large4 as ml4
from torch.nn.attention import sdpa_kernel
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev, dt = "cuda", torch.bfloat16
native = v.native_from_shard(Path(hf_hub_download(ml4.SOURCE_REPO, v.SHARD, revision=ml4.SOURCE_REVISION)))
params = ml4.source_json("params.json")
vision, layers = v.import_reference(v.fetch_reference(Path(sys.argv[1])))
layers.memory_efficient_attention = v.efficient_attention
from mistral_inference.rope import apply_rotary_emb
ref = v.build_reference(vision, layers, params, native, dt, dev)
tower, _ = ml4.load_bundle_parts(ml4.RELEASE_DIR, dt)
tower = tower.to(dev)
out = {}
for res, name in ((448, "validation_006500.jpg"), (448, "validation_000000.jpg")):
    pixels = ml4.Preprocess(res)(Image.open(f"{sys.argv[2]}/{name}")).to(dev, dt)
    side = res // 14
    with torch.inference_mode():
        enc = ref.vision_encoder
        xr = enc.ln_pre(enc.patch_conv(pixels[None]).squeeze(0).flatten(1).permute(1, 0))
        xa = tower.ln_pre(tower.patch_conv(pixels[None]).flatten(2).transpose(1, 2))
        pos = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1).reshape(-1, 2).to(dev)
        freqs = enc.freqs_cis[pos[:, 0], pos[:, 1]]
        table = ml4.rotary_table(side, side, 128, 10000.0).to(dev)
        rows = []
        first = None
        for i in range(30):
            br, ba = enc.transformer.layers[i], tower.transformer.layers[i]
            yr = br(xr, freqs_cis=freqs)
            ya = ml4.block_forward(ba, xa, table)
            same_in = torch.equal(xa[0], xr)
            # same input fed to the adapter block
            ya_same = ml4.block_forward(ba, xr[None], table)
            rows.append({"layer": i + 1, "inputs_identical": same_in, "chained_exact_fraction": (ya[0] == yr).float().mean().item(),
                         "same_input_bit_exact": bool(torch.equal(ya_same[0], yr)),
                         "same_input_exact_fraction": (ya_same[0] == yr).float().mean().item(),
                         "max_abs_input": xr.abs().max().item()})
            if first is None and not torch.equal(ya_same[0], yr):
                first = i
                # op trace at this layer with identical inputs
                att = br.attention
                n = br.attention_norm(xr)
                na = ba.attention_norm(xr[None])[0]
                xq, xk, xv = att.wq(n).view(-1, 16, 128), att.wk(n).view(-1, 16, 128), att.wv(n).view(-1, 16, 128)
                rq, rk = apply_rotary_emb(xq, xk, freqs_cis=freqs)
                def heads(p): return p(n[None]).view(1, -1, 16, 128).transpose(1, 2)
                q_a = ml4.rotate(heads(ba.attention.q_proj), table)
                k_a = ml4.rotate(heads(ba.attention.k_proj), table)
                v_a = heads(ba.attention.v_proj)
                o_r = v.efficient_attention(rq[None], rk[None], xv[None], None)
                with sdpa_kernel(ml4.ATTENTION_BACKENDS):
                    o_a = torch.nn.functional.scaled_dot_product_attention(q_a, k_a, v_a, scale=att.scale)
                o_a_from_ref = None
                with sdpa_kernel(ml4.ATTENTION_BACKENDS):
                    o_a_from_ref = torch.nn.functional.scaled_dot_product_attention(rq[None].transpose(1, 2), rk[None].transpose(1, 2), xv[None].transpose(1, 2), scale=att.scale)
                trace = {
                    "attention_norm": bool(torch.equal(na, n)),
                    "q_rotated": bool(torch.equal(q_a[0].transpose(0, 1), rq)),
                    "k_rotated": bool(torch.equal(k_a[0].transpose(0, 1), rk)),
                    "v": bool(torch.equal(v_a[0].transpose(0, 1), xv)),
                    "attention_out_adapter_vs_ref": (o_a.transpose(1, 2)[0] == o_r[0]).float().mean().item(),
                    "attention_same_inputs_adapter_call_vs_ref_call": (o_a_from_ref.transpose(1, 2)[0] == o_r[0]).float().mean().item(),
                    "q_strides_adapter": list(q_a.stride()), "q_strides_ref": list(rq[None].transpose(1, 2).stride()),
                    "v_strides_adapter": list(v_a.stride()), "v_strides_ref": list(xv[None].transpose(1, 2).stride()),
                }
                rows[-1]["trace"] = trace
            xr, xa = yr, ya
        out[f"{res}/{name}"] = {"first_non_identical_layer_same_input": None if first is None else first + 1, "layers": rows}
Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + "\n")
for k, o in out.items():
    print(k, "first same-input mismatch:", o["first_non_identical_layer_same_input"])
    for r in o["layers"][:12]:
        print("  ", r["layer"], r["inputs_identical"], round(r["chained_exact_fraction"], 3), r["same_input_bit_exact"], round(r["same_input_exact_fraction"], 5), round(r["max_abs_input"], 1), r.get("trace", ""))
