#!/usr/bin/env python3
"""Measure how far the independent Mistral reference's own BF16 forward drifts from FP32.

Reports, per Stage, the largest activations, the top channels' share of the
energy, and the BF16 error with and without those channels. This explains the
BF16 floor that scripts/validate_mistral_large4.py compares the release with.
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from scripts import validate_mistral_large4 as validate
from vtb.adapters import mistral_large4 as ml4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    native = validate.native_from_shard(Path(hf_hub_download(ml4.SOURCE_REPO, validate.SHARD, revision=ml4.SOURCE_REVISION)))
    params = ml4.source_json("params.json")
    vision, layers = validate.import_reference(validate.fetch_reference(args.reference_root))
    pixels = ml4.Preprocess(args.resolution)(Image.open(args.image))
    depths = tuple(round(30 * (k + 1) / 8) for k in range(8))
    outputs = {}
    for dtype in (torch.float32, torch.bfloat16):
        reference = validate.build_reference(vision, layers, params, native, dtype, args.device)
        with torch.inference_mode():
            stages = reference(pixels.to(args.device, dtype), depths)
        outputs[dtype] = {name: tensor.float().cpu() for name, tensor in stages.items()}
        del reference
        torch.cuda.empty_cache()

    report = {}
    for stage, exact in outputs[torch.float32].items():
        rounded = outputs[torch.bfloat16][stage]
        rms = exact.square().mean(0).sqrt()
        order = rms.argsort(descending=True)
        rest = order[4:]
        token_cosine = torch.nn.functional.cosine_similarity(rounded, exact, dim=-1)
        token_norm = exact.norm(dim=-1)
        report[stage] = {
            "max_abs": exact.abs().max().item(),
            "top_channels": order[:4].tolist(),
            "top_channel_rms": [round(rms[c].item(), 2) for c in order[:4]],
            "median_channel_rms": rms.median().item(),
            "top4_energy_share": (rms[order[:4]].square().sum() / rms.square().sum()).item(),
            "bf16_relative": ((rounded - exact).norm() / exact.norm()).item(),
            "bf16_relative_without_top4_channels": ((rounded[:, rest] - exact[:, rest]).norm() / exact[:, rest].norm()).item(),
            "token_norm_max_over_median": (token_norm.max() / token_norm.median()).item(),
            "bf16_token_cosine_median": token_cosine.median().item(),
            "bf16_token_cosine_min": token_cosine.min().item(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "image": str(args.image), "image_sha256": validate.sha256_file(args.image),
        "resolution": args.resolution, "stages": report,
    }, indent=2) + "\n")
    print(json.dumps({k: round(v["bf16_relative"], 4) for k, v in report.items()}))


if __name__ == "__main__":
    main()
