"""Train the dense readouts on a cached run and report one geometry column.

Mirrors probe_run.py, with two differences. The features are full patch tokens laid back
onto their grid rather than a pooled vector, and the head predicts a map, so predictions
get resized to the target before the loss sees them.
"""

import argparse
import json
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


def load_targets(path: Path, image_ids: list[str], task: str):
    """Read targets written by the dataset prep step, ordered to match the cache.

    The prep step writes one npz holding a target per image id, and a validity mask per
    image id for normals. Keeping targets keyed by image id rather than by position means
    the cache and the labels cannot silently drift out of order.
    """
    store = np.load(path)
    targets = torch.from_numpy(np.stack([store[f"{i}"] for i in image_ids])).float()
    if task == "normal":
        valid = torch.from_numpy(np.stack([store[f"{i}_valid"] for i in image_ids])).float()
        return targets, valid
    return targets, None


def train_cell(features, targets, valid, split, task, args):
    torch.manual_seed(args.seed)
    head_class = geometry.DepthHead if task == "depth" else geometry.SurfaceNormalHead
    model = head_class([features.shape[1]], head=args.head).to(args.device)
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
def score(model, features, targets, valid, index, task, args) -> dict[str, float]:
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
    return {key: round(torch.cat(v).mean().item(), 4) for key, v in totals.items()}


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
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cells = []
    for stage, layer in cache.slices(args.run):
        batch = cache.load_batch(args.run, stage, layer)
        targets, valid = load_targets(args.targets, batch.image_ids, args.task)
        split = split_indices(len(batch.image_ids))

        if args.match_capacity:
            reducer = probe.Reducer.fit(batch.tokens[split.train].float(), args.width)
            batch = batch.with_tokens(reducer(batch.tokens.float()))
        features = geometry.dense_map(batch)

        model = train_cell(features, targets, valid, split, args.task, args)
        result = score(model, features, targets, valid, split.test, args.task, args)
        result |= {
            "model_id": batch.model_id,
            "stage": stage,
            "layer_index": layer,
            "relative_depth": batch.relative_depth,
            "token_width": features.shape[1],
            "trainable_parameters": geometry.parameter_count(model),
        }
        cells.append(result)
        headline = "d1" if args.task == "depth" else "mean_deg"
        print(
            f"{stage:>9} L{layer:02d}  depth {result['relative_depth']:.3f}  "
            f"{headline} {result[headline]:.4f}  rmse {result['rmse']:.4f}  "
            f"params {result['trainable_parameters']:,}",
            flush=True,
        )

    tag = "matched" if args.match_capacity else "raw"
    out = args.out or args.run / f"geometry_{args.task}_{args.head}_{tag}.json"
    out.write_text(json.dumps({"task": args.task, "cells": cells}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
