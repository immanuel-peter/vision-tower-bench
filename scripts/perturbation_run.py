#!/usr/bin/env python3
"""Train semantic readouts on clean features and score fixed perturbed features."""

import argparse
import json
import time
from pathlib import Path

import torch

from vtb import cache, probe_run
from vtb.correspondence import paired_bootstrap


@torch.inference_mode()
def correctness(model, features, labels, index, device, batch_size=512) -> torch.Tensor:
    model.eval()
    values = []
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        prediction = model(features[chunk].to(device)).argmax(dim=-1).cpu()
        values.append((prediction == labels[chunk]).float())
    return torch.cat(values)


def condition_details(manifest: dict) -> dict[str, dict]:
    details = {"identity": {"factor": "identity", "level": 0, "value": 0}}
    names = {
        "scale": "scale",
        "occlusion": "occlusion",
        "motion_blur_radius_pixels": "motion_blur",
    }
    for source_name, factor in names.items():
        if source_name not in manifest["factors"]:
            continue
        for level, value in enumerate(manifest["factors"][source_name][1:], 1):
            details[f"{factor}_l{level}"] = {
                "factor": factor,
                "level": level,
                "value": value,
            }
    return details


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-run", type=Path, required=True)
    ap.add_argument("--features-root", type=Path, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--transform-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--readout", default="attention", choices=("attention", "mean"))
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--learning-rates", type=float, nargs="+", default=list(probe_run.LEARNING_RATES))
    ap.add_argument("--resamples", type=int, default=10_000)
    return ap


def main() -> None:
    args = parser().parse_args()
    transform_manifest = json.loads(args.transform_manifest.read_text())
    details = condition_details(transform_manifest)
    conditions = [name for name in transform_manifest["conditions"] if name != "identity"]
    label_map = json.loads(args.labels.read_text())["labels"]
    num_classes = len(set(label_map.values()))

    slices = cache.slices(args.clean_run)
    deepest = max(layer for stage, layer in slices if stage == "tower")
    selected = [cell for cell in slices if cell[1] == deepest]
    cells = []
    for stage, layer in selected:
        started = time.perf_counter()
        tokens, image_ids, meta = cache.load(args.clean_run, stage, layer)
        labels = torch.tensor([label_map[image_id] for image_id in image_ids])
        split = probe_run.split_indices(len(image_ids))
        reducer = probe_run.fit_reducer(tokens.float(), split, args.width)
        clean = reducer(tokens.float())

        search = {}
        best_rate, best_val = None, -1.0
        for rate in args.learning_rates:
            _, val, _ = probe_run.train_once(
                clean, labels, split, num_classes, args.readout, rate, 0,
                args.device, args.epochs, args.batch_size,
            )
            search[f"{rate:g}"] = round(val, 4)
            if val > best_val:
                best_rate, best_val = rate, val

        models = []
        clean_correct = []
        for seed in range(args.seeds):
            model, _, _ = probe_run.train_once(
                clean, labels, split, num_classes, args.readout, best_rate, seed,
                args.device, args.epochs, args.batch_size,
            )
            models.append(model)
            clean_correct.append(correctness(model, clean, labels, split.test, args.device))
        clean_correct = torch.stack(clean_correct)

        condition_results = [{
            **details["identity"],
            "condition": "identity",
            "accuracy": clean_correct.mean().item(),
            "accuracy_by_seed": clean_correct.mean(dim=1).tolist(),
            "clean_minus_condition": {
                "point_estimate": 0.0, "lower": 0.0, "upper": 0.0,
                "confidence": 0.95, "resamples": args.resamples,
                "bootstrap_seed": 0, "difference": "first_minus_second",
            },
        }]
        measurements = {}
        test_ids = [image_ids[index] for index in split.test.tolist()]
        for condition in conditions:
            condition_run = args.features_root / condition / args.model_dir
            perturbed_tokens, perturbed_ids, _ = cache.load(condition_run, stage, layer)
            if perturbed_ids != image_ids:
                raise ValueError(f"image order differs for {condition_run}")
            perturbed = reducer(perturbed_tokens.float())
            perturbed_correct = torch.stack([
                correctness(model, perturbed, labels, split.test, args.device)
                for model in models
            ])
            interval = paired_bootstrap(
                clean_correct.mean(dim=0), perturbed_correct.mean(dim=0),
                resamples=args.resamples,
            )
            condition_results.append({
                **details[condition],
                "condition": condition,
                "accuracy": perturbed_correct.mean().item(),
                "accuracy_by_seed": perturbed_correct.mean(dim=1).tolist(),
                "clean_minus_condition": interval,
            })
            measurements[condition] = [
                {
                    "image_id": image_id,
                    "clean": clean_correct[:, position].tolist(),
                    "perturbed": perturbed_correct[:, position].tolist(),
                }
                for position, image_id in enumerate(test_ids)
            ]

        cells.append({
            "model_id": meta["model_id"],
            "stage": stage,
            "layer_index": layer,
            "relative_depth": float(meta["relative_depth"]),
            "capacity_matched": True,
            "readout": args.readout,
            "learning_rate": best_rate,
            "learning_rate_search": search,
            "validation_accuracy": best_val,
            "train_images": len(split.train),
            "validation_images": len(split.val),
            "test_images": len(split.test),
            "conditions": condition_results,
            "measurements": measurements,
            "seconds": round(time.perf_counter() - started, 1),
        })
        print(f"{stage}:{layer} clean {clean_correct.mean().item():.4f} across {len(conditions)} perturbations")

    output = {
        "model": args.model_dir,
        "images": transform_manifest["images"],
        "protocol": {
            "training": "clean features only",
            "inference": "same frozen reducer and readout for every condition",
            "capacity_matched_width": args.width,
            "readout": args.readout,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "learning_rates": args.learning_rates,
            "pairing": "seed-mean correctness, paired over test images",
        },
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
