"""Retain every training-channel statistic behind the isolated Qwen diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from vtb import cache, probe_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and torch.get_num_threads() == 6
    rows = []
    for model in ["qwen38_flash_next", "qwen3_5"]:
        run = args.features_root / f"{model}_448_pool4"
        for stage in ["tower", "merged", "projected"]:
            tokens, ids, metadata = cache.load(run, stage, 27)
            split = probe_run.split_indices(len(ids))
            assert len(ids) == 2000 and len(split.train) == 1400
            flat = tokens[split.train].reshape(-1, tokens.shape[-1]).float()
            mean = flat.mean(0)
            sample_std = flat.std(0)
            scale = sample_std.clamp_min(1e-6)
            rms = flat.double().square().mean(0).sqrt()
            assert all(torch.isfinite(t).all() for t in [mean, sample_std, scale, rms])
            rows.append({
                "model": model, "model_id": metadata["model_id"], "stage": stage, "layer": 27,
                "fit_images": len(split.train), "tokens_per_image": tokens.shape[1],
                "fit_image_ids": [ids[i] for i in split.train.tolist()],
                "mean_fp32": mean.tolist(), "sample_std_fp32": sample_std.tolist(),
                "scale_fp32": scale.tolist(), "rms_fp64": rms.tolist(),
                "cache_shards_sha256": {
                    str(path.relative_to(args.features_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in cache.shards(run, stage, 27)},
            })
            del tokens, flat, mean, sample_std, scale, rms
    output = {"fit": "clean training split only", "std_correction": 1, "minimum_scale": 1e-6,
              "torch_threads": 6, "device": "cpu", "torch": torch.__version__,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
