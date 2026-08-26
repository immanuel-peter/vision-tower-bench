"""Train the dense readouts on a cached run and report one geometry column.

Mirrors probe_run.py, with two differences. The features are full patch tokens laid back
onto their grid rather than a pooled vector, and the head predicts a map, so predictions
get resized to the target before the loss sees them.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import interpolate

from vtb import cache, geometry, probe
from vtb.geometry_metrics import (
    depth_si_loss,
    evaluate_depth,
    evaluate_surface_normal,
    normal_loss,
)
from vtb.probe_run import split_indices

TASKS = ("depth", "normal")


def manifest(targets: Path) -> dict:
    """The prep step's manifest, which sits beside the targets it describes."""
    path = targets.with_name(targets.name.replace("_targets.npz", "_manifest.json"))
    return json.loads(path.read_text()) if path.exists() else {}


def depth_range(targets: Path, override: float | None) -> float:
    """Depth bin ceiling, from the flag, else the manifest the prep step wrote."""
    if override:
        return override
    return float(manifest(targets).get("max_depth_metres") or 10.0)


def centre_square(target):
    """Crop a target to the square the Tower was actually shown.

    vtb.images.square_crop resizes the short side and centre-crops, so a 768 by 1024
    DIODE frame reaches every Stage as its middle 768 by 768. Scoring the full frame
    would charge each cell for a quarter of the pixels no Stage ever saw.
    """
    height, width = target.shape[-2:]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return target[..., top:top + side, left:left + side]


def load_targets(path: Path, image_ids: list[str], task: str):
    """Read targets written by the dataset prep step, ordered to match the cache.

    The prep step writes one npz holding a target per image id, and a validity mask per
    image id for normals. Keeping targets keyed by image id rather than by position means
    the cache and the labels cannot silently drift out of order.
    """
    store = np.load(path)

    def stack(suffix: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([centre_square(store[f"{i}{suffix}"]) for i in image_ids])).float()

    if task == "normal":
        return stack("_normal"), stack("_valid").unsqueeze(1)
    return stack("").unsqueeze(1), None


def coverage_of(targets: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """Fraction of each target the metric can score, which the writeup reports by scene.

    DIODE ships no validity mask for normals, so an unannotated pixel is a zero vector
    and coverage runs far lower outdoors than indoors.
    """
    mask = valid if valid is not None else (targets > 0).float()
    return mask.flatten(1).mean(dim=1)


def train_cell(features, targets, valid, split, task, args):
    torch.manual_seed(args.seed)
    if task == "depth":
        model = geometry.DepthHead([features.shape[1]], head=args.head, max_depth=args.max_depth)
    else:
        model = geometry.SurfaceNormalHead([features.shape[1]], head=args.head)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    for _ in range(args.epochs):
        model.train()
        order = split.train[torch.randperm(len(split.train))]
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            loss = _loss(model, features, targets, valid, index, task, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()

    return model


def _predict(model, features, index, device, target_hw):
    prediction = model([features[index].to(device)])
    return interpolate(prediction, target_hw, mode="bilinear", align_corners=False)


def _loss(model, features, targets, valid, index, task, device):
    batch = targets[index].to(device)
    prediction = _predict(model, features, index, device, batch.shape[-2:])
    if task == "depth":
        return depth_si_loss(prediction, batch)
    return normal_loss(prediction, batch, valid[index].to(device))


@torch.inference_mode()
def score(model, features, targets, valid, index, task, args) -> dict[str, torch.Tensor]:
    """Metrics for each test image, left unaveraged so the caller can split by scene."""
    model.eval()
    totals: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(index), args.batch_size):
        chunk = index[start:start + args.batch_size]
        batch = targets[chunk].to(args.device)
        prediction = _predict(model, features, chunk, args.device, batch.shape[-2:])
        if task == "depth":
            metrics = evaluate_depth(prediction, batch, scale_invariant=args.scale_invariant)
        else:
            metrics = evaluate_surface_normal(prediction, batch, valid[chunk].to(args.device))
        for key, value in metrics.items():
            totals.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(v) for key, v in totals.items()}


def summarise(metrics, coverage, image_ids, index, scenes) -> dict:
    """Average every metric over the test split, then again within each scene type."""
    rows: dict[str, list[int]] = {}
    for position, image_id in enumerate(index.tolist()):
        rows.setdefault(scenes.get(image_ids[image_id], "unknown"), []).append(position)

    def block(chosen: torch.Tensor) -> dict:
        out = {key: round(value[chosen].mean().item(), 4) for key, value in metrics.items()}
        return out | {"images": len(chosen), "coverage": round(coverage[chosen].mean().item(), 4)}

    everything = block(torch.arange(len(index)))
    return everything | {
        "by_scene": {name: block(torch.tensor(sorted(r))) for name, r in sorted(rows.items())}
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the dense readouts on a cached run.")
    ap.add_argument("--run", type=Path, required=True, help="full-token cache from vtb.extract")
    ap.add_argument("--targets", type=Path, required=True, help="npz written by the dataset prep step")
    ap.add_argument("--task", default="depth", choices=TASKS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--head", default="multiscale", choices=("multiscale", "linear"))
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument(
        "--match-capacity",
        action="store_true",
        help="fit a frozen PCA reduction to --width first. A multiscale depth head reads "
        "1.71M parameters at a 1024-wide Stage and 4.85M at 7168 (ADR-0008, ADR-0010)",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-invariant", action="store_true")
    ap.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="depth bin ceiling in metres. Read from the prep manifest when omitted, "
        "because DIODE reaches 230 m outdoors against NYU's 10 m indoors.",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.max_depth = depth_range(args.targets, args.max_depth)
    scenes = manifest(args.targets).get("scenes", {})
    print(f"depth bins span 0 to {args.max_depth:.1f} m")

    cells = []
    order, targets, valid, coverage = None, None, None, None
    for stage, layer in cache.slices(args.run):
        started = time.perf_counter()
        batch = cache.load_batch(args.run, stage, layer)
        # Every slice of a run holds the same images in the same order, so the 6.3 GB of
        # targets are read once rather than once per cell.
        if batch.image_ids != order:
            order = batch.image_ids
            targets, valid = load_targets(args.targets, order, args.task)
            coverage = coverage_of(targets, valid)
        split = split_indices(len(order))

        if args.match_capacity:
            reducer = probe.Reducer.fit(batch.tokens[split.train].float(), args.width)
            batch = batch.with_tokens(reducer(batch.tokens.float()))
        # The cache holds bfloat16; the heads are float32.
        features = geometry.dense_map(batch).float()

        model = train_cell(features, targets, valid, split, args.task, args)
        metrics = score(model, features, targets, valid, split.test, args.task, args)
        result = summarise(metrics, coverage, order, split.test, scenes)
        result |= {
            "max_depth": args.max_depth,
            "model_id": batch.model_id,
            "stage": stage,
            "layer_index": layer,
            "relative_depth": batch.relative_depth,
            "token_width": features.shape[1],
            "grid": features.shape[-1],
            "trainable_parameters": geometry.parameter_count(model),
            "seconds": round(time.perf_counter() - started, 1),
        }
        cells.append(result)
        headline = "d1" if args.task == "depth" else "mean_deg"
        print(
            f"{stage:>9} L{layer:02d}  depth {result['relative_depth']:.3f}  "
            f"{headline} {result[headline]:.4f}  rmse {result['rmse']:.4f}  "
            f"params {result['trainable_parameters']:,}  {result['seconds']:.0f}s",
            flush=True,
        )

    tag = "matched" if args.match_capacity else "raw"
    out = args.out or args.run / f"geometry_{args.task}_{args.head}_{tag}.json"
    out.write_text(
        json.dumps(
            {"task": args.task, "images": len(order), "train_images": len(split.train),
             "capacity_matched": args.match_capacity, "head": args.head,
             "epochs": args.epochs, "seed": args.seed, "cells": cells},
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
