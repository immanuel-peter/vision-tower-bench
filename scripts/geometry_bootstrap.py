#!/usr/bin/env python3
"""Train two selected geometry cells and write a paired image-bootstrap dataset."""

import argparse
import gc
import json
from pathlib import Path

import torch

from vtb import cache, geometry, geometry_run, probe_run
from vtb.geometry_bootstrap import paired_metric_bootstrap


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-run", type=Path, required=True)
    ap.add_argument("--first-name", required=True)
    ap.add_argument("--first-stage", choices=("tower", "merged", "projected"), required=True)
    ap.add_argument("--first-layer", type=int, required=True)
    ap.add_argument("--second-run", type=Path, required=True)
    ap.add_argument("--second-name", required=True)
    ap.add_argument("--second-stage", choices=("tower", "merged", "projected"), required=True)
    ap.add_argument("--second-layer", type=int, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", choices=geometry_run.TASKS, required=True)
    ap.add_argument("--arm", choices=("raw", "matched"), default="matched")
    ap.add_argument("--head", default="multiscale", choices=("multiscale", "linear"))
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scale-invariant", action="store_true")
    ap.add_argument("--max-depth", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--learning-rates", type=float, nargs="+", default=list(geometry_run.LEARNING_RATES)
    )
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    return ap


def train_cell(run: Path, stage: str, layer: int, args) -> dict:
    batch = cache.load_batch(run, stage, layer)
    image_ids = batch.image_ids
    split = probe_run.split_indices(len(image_ids))
    targets, valid = geometry_run.load_targets(args.targets, image_ids, args.task)

    if args.arm == "matched":
        batch = geometry_run.match_capacity(batch, split, args.width)
    features = geometry.dense_map(batch).float()

    selected_rate, selection = geometry_run.select_learning_rate(
        features, targets, valid, split, args.task, args
    )
    metric, higher_is_better = geometry_run.SELECTION[args.task]
    values = []
    for seed in range(args.seeds):
        model = geometry_run.train_cell(
            features, targets, valid, split, args.task, args, selected_rate, seed
        )
        scored = geometry_run.score(
            model, features, targets, valid, split.test, args.task, args
        )
        values.append(scored[metric])

    test_ids = [image_ids[index] for index in split.test.tolist()]
    scenes = geometry_run.manifest(args.targets).get("scenes", {})
    result = {
        "image_ids": test_ids,
        "scenes": [scenes.get(image_id, "unknown") for image_id in test_ids],
        "values": torch.stack(values),
        "model_id": batch.model_id,
        "stage": stage,
        "layer_index": layer,
        "relative_depth": float(batch.relative_depth),
        "metric": metric,
        "higher_is_better": higher_is_better,
        "learning_rate": selected_rate,
        "learning_rate_search": selection["learning_rate_search"],
        "validation_score": selection["val_score"],
    }
    del features, targets, valid, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def interval(first: dict, second: dict, positions: torch.Tensor, args) -> dict:
    return paired_metric_bootstrap(
        first["values"][:, positions],
        second["values"][:, positions],
        higher_is_better=first["higher_is_better"],
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.bootstrap_seed,
    )


def cell_summary(cell: dict, name: str) -> dict:
    seed_metrics = cell["values"].mean(dim=1)
    return {
        "name": name,
        "model_id": cell["model_id"],
        "stage": cell["stage"],
        "layer_index": cell["layer_index"],
        "relative_depth": cell["relative_depth"],
        "learning_rate": cell["learning_rate"],
        "learning_rate_search": cell["learning_rate_search"],
        "validation_score": cell["validation_score"],
        "seed_metrics": seed_metrics.tolist(),
        "mean_metric": seed_metrics.mean().item(),
    }


def main() -> None:
    args = parser().parse_args()
    args.max_depth = geometry_run.depth_range(args.targets, args.max_depth)
    first = train_cell(args.first_run, args.first_stage, args.first_layer, args)
    second = train_cell(args.second_run, args.second_stage, args.second_layer, args)

    if first["image_ids"] != second["image_ids"]:
        raise ValueError("the two cells do not have the same paired test images")
    if first["scenes"] != second["scenes"]:
        raise ValueError("the two cells do not have the same scene labels")
    if first["metric"] != second["metric"]:
        raise ValueError("the two cells do not use the same headline metric")

    all_positions = torch.arange(len(first["image_ids"]))
    by_scene = {}
    for scene in sorted(set(first["scenes"])):
        positions = torch.tensor([i for i, value in enumerate(first["scenes"]) if value == scene])
        by_scene[scene] = interval(first, second, positions, args) | {
            "test_images": len(positions)
        }

    rows = []
    for position, image_id in enumerate(first["image_ids"]):
        rows.append(
            {
                "image_id": image_id,
                "scene": first["scenes"][position],
                args.first_name: first["values"][:, position].tolist(),
                args.second_name: second["values"][:, position].tolist(),
            }
        )

    metric = first["metric"]
    payload = {
        "protocol": {
            "task": args.task,
            "metric": metric,
            "higher_is_better": first["higher_is_better"],
            "arm": args.arm,
            "head": args.head,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "learning_rates": args.learning_rates,
            "test_images": len(rows),
            "pairing": "seed-mean metric, paired over test images",
        },
        "first": cell_summary(first, args.first_name),
        "second": cell_summary(second, args.second_name),
        "first_advantage": interval(first, second, all_positions, args),
        "by_scene": by_scene,
        "measurements": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    summary = {key: value for key, value in payload.items() if key != "measurements"}
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
