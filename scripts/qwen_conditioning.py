"""Audit and isolate channel-standardization sensitivity for the Qwen perturbation readouts."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts import perturbation_run
from scripts.standardized_label_budget import PATCH_SHA256, PATCH_SOURCE, fit_reducer as standardized_fit
from vtb import cache, probe_run

PRIMARY_FIT = probe_run.fit_reducer
PRIMARY_TRAIN = probe_run.train_once


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_with_loss_log(*args, **kwargs):
    losses = []
    original = F.cross_entropy

    def record(logits, labels, *extra, **options):
        loss = original(logits, labels, *extra, **options)
        value = loss.detach().item()
        losses.append({"loss": value if math.isfinite(value) else None, "images": len(labels)})
        return loss

    with patch.object(F, "cross_entropy", record):
        result = PRIMARY_TRAIN(*args, **kwargs)
    split, epochs, batch = args[2], args[8], args[9]
    steps = math.ceil(len(split.train) / batch)
    epoch_losses = []
    for start in range(0, len(losses), steps):
        group = losses[start:start + steps]
        epoch_losses.append(None if any(x["loss"] is None for x in group) else
                            sum(x["loss"] * x["images"] for x in group) / sum(x["images"] for x in group))
    assert len(epoch_losses) == epochs
    return result, {
        "learning_rate": args[5], "seed": args[6], "epochs": epochs, "batch_size": batch,
        "train_images": len(split.train), "step_losses": losses, "epoch_losses": epoch_losses,
        "nonfinite_loss_steps": sum(x["loss"] is None for x in losses),
        "validation_accuracy": result[1], "test_accuracy": result[2],
    }


def statistics(features, split, reducer):
    training = features[split.train]
    flat = training.reshape(-1, training.shape[-1])
    rms = flat.double().square().mean(0).sqrt()
    centered = flat - reducer.mean
    if hasattr(reducer, "scale"):
        centered = centered / reducer.scale
    total = centered.double().var(0).sum()
    reduced = reducer(training).reshape(-1, reducer.basis.shape[1]).double()
    variances = reduced.var(0)
    top = rms.topk(min(16, len(rms)))
    return {
        "fit_images": len(split.train), "tokens_per_image": features.shape[1],
        "native_width": features.shape[-1], "reduced_width": reducer.basis.shape[1],
        "channel_rms_max": rms.max().item(), "channel_rms_median": rms.median().item(),
        "channel_rms_max_to_median": (rms.max() / rms.median()).item(),
        "top_rms_channels": [{"channel": int(i), "rms": float(v)} for i, v in zip(top.indices, top.values)],
        "pca_input_variance": total.item(), "pc1_share": (variances[0] / total).item(),
        "top8_pc_variance_shares": (variances[:8] / total).tolist(),
        "pc1_max_loading_channel": int(reducer.basis[:, 0].abs().argmax()),
        "pc1_max_abs_loading": reducer.basis[:, 0].abs().max().item(),
        "standardized": hasattr(reducer, "scale"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["audit", "sensitivity"], required=True)
    ap.add_argument("--clean-run", type=Path, required=True)
    ap.add_argument("--features-root", type=Path, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--transform-manifest", type=Path, required=True)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert torch.get_num_threads() == 6
    assert args.out.resolve() != args.primary.resolve()
    original_hash = sha(args.primary)
    primary = json.loads(args.primary.read_text())
    assert primary["protocol"]["seeds"] == 3 and primary["protocol"]["epochs"] == 20
    slices = cache.slices(args.clean_run)
    deepest = max(layer for _, layer in slices)
    selected = [(s, l) for s, l in slices if l == deepest]
    assert len(selected) == 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats, training_logs = [], []
    if args.mode == "audit":
        label_map = json.loads(args.labels.read_text())["labels"]
        for stage, layer in selected:
            tokens, ids, meta = cache.load(args.clean_run, stage, layer)
            features = tokens.float()
            split = probe_run.split_indices(len(ids))
            reducer = PRIMARY_FIT(features, split, 512)
            row = statistics(features, split, reducer)
            row.update(stage=stage, layer=layer, model_id=meta["model_id"])
            clean = reducer(features)
            labels = torch.tensor([label_map[i] for i in ids])
            cell = next(c for c in primary["cells"] if c["stage"] == stage)
            result, loss = train_with_loss_log(clean, labels, split, len(set(label_map.values())),
                                               "attention", cell["learning_rate"], 0, "cpu", 20, 256)
            expected = cell["conditions"][0]["accuracy_by_seed"][0]
            assert abs(result[2] - expected) < 1e-6, (stage, result[2], expected)
            row["primary_seed0_replay_accuracy"] = result[2]
            row["primary_recorded_seed0_accuracy"] = expected
            loss.update(stage=stage, phase="selected_primary_seed0_replay")
            stats.append(row)
            training_logs.append(loss)
            print(stage, "primary replay matched; PC1 share", row["pc1_share"], flush=True)
        output = {"mode": "audit", "statistics": stats, "training_losses": training_logs}
    else:
        current = {"stage": None}
        counts = {}

        def fit(features, split, width):
            stage, layer = selected[len(stats)]
            current["stage"] = stage
            reducer = standardized_fit(features, split, width)
            row = statistics(features, split, reducer)
            row.update(stage=stage, layer=layer)
            stats.append(row)
            print(stage, "standardized PCA fitted", flush=True)
            return reducer

        def train(*parameters, **options):
            result, loss = train_with_loss_log(*parameters, **options)
            stage = current["stage"]
            index = counts.get(stage, 0)
            loss.update(stage=stage, phase="search" if index < 11 else "selected", call=index)
            counts[stage] = index + 1
            training_logs.append(loss)
            return result

        forwarded = ["conditioning", "--clean-run", str(args.clean_run), "--features-root", str(args.features_root),
                     "--model-dir", args.model_dir, "--labels", str(args.labels),
                     "--transform-manifest", str(args.transform_manifest), "--out", str(args.out),
                     "--device", "cpu", "--epochs", "20", "--seeds", "3", "--width", "512", "--batch-size", "256"]
        with patch.object(sys, "argv", forwarded), patch.object(probe_run, "fit_reducer", fit), patch.object(probe_run, "train_once", train):
            perturbation_run.main()
        output = json.loads(args.out.read_text())
        assert len(stats) == 3 and len(training_logs) == 42 and set(counts.values()) == {14}
        output["protocol"]["reducer"] = "training-only sample-standardized PCA"
        output["protocol"]["reducer_patch_sha256"] = PATCH_SHA256
        output["conditioning_statistics"] = stats
        output["training_losses"] = training_logs
        output["diagnostic"] = {"changed_component": "reducer_only", "patch_source": PATCH_SOURCE,
                                "patch_sha256": PATCH_SHA256}
    assert sha(args.primary) == original_hash
    output["provenance"] = {
        "primary_source": str(args.primary), "primary_sha256_before_and_after": original_hash,
        "script_sha256": sha(__file__), "primary_probe_source_sha256": sha(Path(probe_run.__file__)),
        "labels_sha256": sha(args.labels), "transform_manifest_sha256": sha(args.transform_manifest),
        "device": "cpu", "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
        "fit_split": "clean training images only; seed0 split and PCA", "mean_std_correction": 1,
    }
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

