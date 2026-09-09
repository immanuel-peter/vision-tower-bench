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

from vtb import cache, geometry
from vtb.geometry_metrics import (
    depth_si_loss,
    evaluate_depth,
    evaluate_surface_normal,
    normal_loss,
)
from vtb.probe_run import fit_reducer, split_indices

TASKS = ("depth", "normal")

# Validation metric and direction for each task.
SELECTION = {"depth": ("d1", True), "normal": ("mean_deg", False)}

# Six points spanning both edges the four-point matrix truncated at.
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def manifest(targets: Path) -> dict:
    """Read the prep manifest beside the target archive."""
    path = targets.with_name(targets.name.replace("_targets.npz", "_manifest.json"))
    return json.loads(path.read_text()) if path.exists() else {}


def depth_range(targets: Path, override: float | None) -> float:
    """Depth bin ceiling, from the flag, else the manifest the prep step wrote."""
    if override:
        return override
    return float(manifest(targets).get("max_depth_metres") or 10.0)


def selected_slices(
    run: Path,
    only: list[str] | None = None,
    stages: list[str] | None = None,
    deepest_only: bool = False,
) -> list[tuple[str, int]]:
    chosen = cache.slices(run)
    if only:
        wanted = {(value.split(":")[0], int(value.split(":")[1])) for value in only}
        chosen = [cell for cell in chosen if cell in wanted]
        if len(chosen) != len(wanted):
            raise ValueError(f"{only} does not match slices in {run}")
    if stages:
        wanted_stages = set(stages)
        chosen = [cell for cell in chosen if cell[0] in wanted_stages]
    if deepest_only:
        deepest = max(layer for stage, layer in cache.slices(run) if stage == "tower")
        chosen = [cell for cell in chosen if cell[1] == deepest]
    return chosen


def centre_square(target):
    """Crop a target to the field of view passed through square_crop."""
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
    """Treat zero vectors as unannotated when no validity mask exists."""
    mask = valid if valid is not None else (targets > 0).float()
    return mask.flatten(1).mean(dim=1)


def train_cell(features, targets, valid, split, task, args, learning_rate, seed):
    torch.manual_seed(seed)
    model = build_head(features.shape[1], task, args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
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
    # Reorder image-keyed coverage to match test-split metrics.
    coverage = coverage[index]

    def block(chosen: torch.Tensor) -> dict:
        out = {key: round(value[chosen].mean().item(), 4) for key, value in metrics.items()}
        return out | {"images": len(chosen), "coverage": round(coverage[chosen].mean().item(), 4)}

    everything = block(torch.arange(len(index)))
    return everything | {
        "by_scene": {name: block(torch.tensor(sorted(r))) for name, r in sorted(rows.items())}
    }


def select_learning_rate(features, targets, valid, split, task, args) -> tuple[float, dict]:
    """Train once per rate and keep the one that scores best on the validation split."""
    metric, higher_is_better = SELECTION[task]
    searched: dict[str, float] = {}
    best_rate, best_score = None, None
    for rate in args.learning_rates:
        model = train_cell(features, targets, valid, split, task, args, rate, seed=0)
        value = score(model, features, targets, valid, split.val, task, args)[metric].mean().item()
        searched[f"{rate:g}"] = round(value, 4)
        if best_score is None or (value > best_score if higher_is_better else value < best_score):
            best_rate, best_score = rate, value
    return best_rate, {
        "learning_rate": best_rate,
        "val_metric": metric,
        "val_score": round(best_score, 4),
        "learning_rate_search": searched,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    stacked = torch.tensor(values, dtype=torch.float64)
    return round(stacked.mean().item(), 4), round(stacked.std(unbiased=False).item(), 4)


def aggregate(runs: list[dict]) -> dict:
    """Leave split metadata unchanged while aggregating fitted metrics over seeds."""
    out: dict = {}
    for key, value in runs[0].items():
        if key == "by_scene":
            out[key] = {name: aggregate([r["by_scene"][name] for r in runs]) for name in value}
        elif key in ("images", "coverage"):
            out[key] = value
        else:
            out[key], out[f"{key}_std"] = _mean_std([r[key] for r in runs])
    return out


def build_head(width: int, task: str, args) -> torch.nn.Module:
    if task == "depth":
        return geometry.DepthHead([width], head=args.head, max_depth=args.max_depth)
    return geometry.SurfaceNormalHead([width], head=args.head)


def match_capacity(batch, split, width: int):
    """Apply the same deterministic, train-split-only reduction as semantics."""
    reducer = fit_reducer(batch.tokens.float(), split, width)
    return batch.with_tokens(reducer(batch.tokens.float()))


def run_cell(features, targets, valid, split, task, coverage, image_ids, scenes, args) -> dict:
    best_rate, selection = select_learning_rate(features, targets, valid, split, task, args)
    per_seed = []
    for seed in range(args.seeds):
        model = train_cell(features, targets, valid, split, task, args, best_rate, seed)
        metrics = score(model, features, targets, valid, split.test, task, args)
        per_seed.append(summarise(metrics, coverage, image_ids, split.test, scenes))
    return aggregate(per_seed) | selection | {"seeds": args.seeds, "per_seed": per_seed}


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
        help="reduce each Stage to --width before training the head",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(LEARNING_RATES),
        help="search these rates on validation for every cell",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scale-invariant", action="store_true")
    ap.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="depth ceiling in metres; defaults to the prep manifest",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STAGE:LAYER",
        help="run only the given STAGE:LAYER cells",
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        choices=("tower", "merged", "projected"),
        default=None,
        help="run only these Stages",
    )
    ap.add_argument(
        "--deepest-only",
        action="store_true",
        help="run only cells at the final Tower layer",
    )
    args = ap.parse_args()
    args.max_depth = depth_range(args.targets, args.max_depth)
    scenes = manifest(args.targets).get("scenes", {})
    print(f"depth bins span 0 to {args.max_depth:.1f} m")

    try:
        chosen = selected_slices(args.run, args.only, args.stages, args.deepest_only)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    cells = []
    order, targets, valid, coverage = None, None, None, None
    for stage, layer in chosen:
        started = time.perf_counter()
        batch = cache.load_batch(args.run, stage, layer)
        # Reuse the 6.3 GB target array while image order stays unchanged.
        if batch.image_ids != order:
            order = batch.image_ids
            targets, valid = load_targets(args.targets, order, args.task)
            coverage = coverage_of(targets, valid)
        split = split_indices(len(order))

        if args.match_capacity:
            batch = match_capacity(batch, split, args.width)
        # The cache holds bfloat16; the heads are float32.
        features = geometry.dense_map(batch).float()

        parameters = geometry.parameter_count(build_head(features.shape[1], args.task, args))
        result = run_cell(
            features, targets, valid, split, args.task, coverage, order, scenes, args
        )
        result |= {
            "max_depth": args.max_depth,
            "model_id": batch.model_id,
            "stage": stage,
            "layer_index": layer,
            "relative_depth": batch.relative_depth,
            "token_width": features.shape[1],
            "grid": features.shape[-1],
            "trainable_parameters": parameters,
            "seconds": round(time.perf_counter() - started, 1),
        }
        cells.append(result)
        headline = SELECTION[args.task][0]
        print(
            f"{stage:>9} L{layer:02d}  depth {result['relative_depth']:.3f}  "
            f"{headline} {result[headline]:.4f} +- {result[headline + '_std']:.4f}  "
            f"lr {result['learning_rate']:g}  params {parameters:,}  "
            f"{result['seconds']:.0f}s",
            flush=True,
        )

    tag = "matched" if args.match_capacity else "raw"
    out = args.out or args.run / f"geometry_{args.task}_{args.head}_{tag}.json"

    out.write_text(
        json.dumps(
            {"task": args.task, "images": len(order), "train_images": len(split.train),
             "capacity_matched": args.match_capacity, "head": args.head,
             "epochs": args.epochs, "seeds": args.seeds,
             "learning_rates": args.learning_rates, "cells": cells},
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
