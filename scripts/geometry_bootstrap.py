#!/usr/bin/env python3
"""Train two selected geometry cells and write a paired image-bootstrap dataset."""

import argparse
import gc
import hashlib
import json
import os
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
    ap.add_argument("--cell-cache", type=Path, default=os.environ.get("VTB_GEOMETRY_PAIRED_CACHE"),
                    help="optional cache of verified paired-rerun cells; primary training protocol is unchanged")
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


def cached_train_cell(run: Path, stage: str, layer: int, args) -> dict:
    if args.cell_cache is None:
        return train_cell(run, stage, layer, args)
    protocol = {key: getattr(args, key) for key in (
        "task", "arm", "head", "width", "epochs", "batch_size", "seeds",
        "scale_invariant", "max_depth", "learning_rates", "device",
    )}
    paths = [args.targets, Path(__file__)] + list(cache.shards(run, stage, layer))
    manifest = args.targets.with_name(args.targets.name.replace("_targets.npz", "_manifest.json"))
    if manifest.is_file():
        paths.append(manifest)
    paths += sorted((Path(__file__).resolve().parents[1] / "vtb").rglob("*.py"))
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    identity = {"stage": stage, "layer": layer, "protocol": protocol,
                "files": hashes, "torch": str(torch.__version__),
                "threads": torch.get_num_threads(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.cell_cache.mkdir(parents=True, exist_ok=True)
    path = args.cell_cache / f"{key}.pt"
    if path.is_file():
        stored = torch.load(path, map_location="cpu", weights_only=True)
        assert stored["identity"] == identity
        result = stored["result"]
        assert result["values"].shape == (args.seeds, len(result["image_ids"]))
        assert torch.isfinite(result["values"]).all()
        print(f"reuse verified paired cell {stage}:{layer} {key}", flush=True)
        return result
    result = train_cell(run, stage, layer, args)
    assert torch.isfinite(result["values"]).all()
    result["paired_cache_key"] = key
    scratch = path.with_suffix(f".{os.getpid()}.partial")
    torch.save({"identity": identity, "result": result}, scratch)
    scratch.replace(path)
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
    } | ({"paired_cache_key": cell["paired_cache_key"]} if "paired_cache_key" in cell else {})


def main() -> None:
    args = parser().parse_args()
    args.max_depth = geometry_run.depth_range(args.targets, args.max_depth)
    first = cached_train_cell(args.first_run, args.first_stage, args.first_layer, args)
    second = cached_train_cell(args.second_run, args.second_stage, args.second_layer, args)

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
