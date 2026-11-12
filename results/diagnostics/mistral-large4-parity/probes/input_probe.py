"""Capture the actual block-1 input inside both full forwards and compare."""
import json, sys
from pathlib import Path
import torch
from PIL import Image
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
tower, _ = ml4.load_bundle_parts(ml4.RELEASE_DIR, dt)
tower = tower.to(dev)
pixels = ml4.Preprocess(448)(Image.open(sys.argv[2] + "/validation_006500.jpg")).to(dev, dt)
cap = {}
h1 = ref.vision_encoder.transformer.layers[0].register_forward_pre_hook(lambda m, a, k=None: cap.__setitem__("ref_in", a[0]))
h2 = tower.transformer.layers[0].register_forward_pre_hook(lambda m, a: None)
with torch.inference_mode():
    ref(pixels, (30,))
    embeds = tower.patch_conv(pixels[None])
    x3 = tower.ln_pre(embeds.flatten(2).transpose(1, 2))
    x3c = tower.ln_pre(embeds.flatten(2).transpose(1, 2).contiguous())
    enc = ref.vision_encoder
    pe = enc.patch_conv(pixels[None]).squeeze(0)
    pre_cat = pe.flatten(1).permute(1, 0)
    after_cat = torch.cat([pre_cat], dim=0)
out = {
    "ref_block1_input_strides": list(cap["ref_in"].stride()),
    "adapter_ln_pre_strides": list(x3.stride()),
    "adapter_strided_vs_ref_exact_fraction": (x3[0] == cap["ref_in"]).float().mean().item(),
    "adapter_contiguous_vs_ref_exact_fraction": (x3c[0] == cap["ref_in"]).float().mean().item(),
    "torch_cat_output_strides": list(after_cat.stride()),
    "ln_pre_strided_vs_contiguous_exact_fraction_ref": (enc.ln_pre(pre_cat) == enc.ln_pre(after_cat)).float().mean().item(),
}
print(json.dumps(out, indent=1))
Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + "\n")
