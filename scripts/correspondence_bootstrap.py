#!/usr/bin/env python3
"""Write a paired image-pair bootstrap interval for two correspondence Stages."""

import argparse
import json
import math
from pathlib import Path

import torch

from vtb.correspondence import paired_bootstrap

PRIMARY = {"scannet": "recall_10px", "navi": "recall_2cm", "spair": "pck_0.1"}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--first-stage", default="projected", choices=("tower", "merged", "projected"))
    ap.add_argument("--second-stage", default="tower", choices=("tower", "merged", "projected"))
    ap.add_argument("--metric", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    return ap


def cell_key(payload: dict, stage: str) -> str:
    matches = [f"{cell['stage']}:{cell['layer_index']}" for cell in payload["cells"] if cell["stage"] == stage]
    if len(matches) != 1:
        raise ValueError(f"expected one {stage} cell, found {matches}")
    return matches[0]


def interval(rows, first_key, second_key, metric, args):
    paired = [
        (row["cells"][first_key][metric], row["cells"][second_key][metric])
        for row in rows
    ]
    paired = [(first, second) for first, second in paired if math.isfinite(first) and math.isfinite(second)]
    result = paired_bootstrap(
        torch.tensor([first for first, _ in paired]),
        torch.tensor([second for _, second in paired]),
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.bootstrap_seed,
    )
    return result | {"pairs": len(paired)}


def main() -> None:
    args = parser().parse_args()
    payload = json.loads(args.run.read_text())
    metric = args.metric or PRIMARY[payload["dataset"]]
    first_key = cell_key(payload, args.first_stage)
    second_key = cell_key(payload, args.second_stage)
    rows = payload["measurements"]
    result = interval(rows, first_key, second_key, metric, args)

    by_difficulty = {}
    difficulty_key = "viewpoint" if payload["dataset"] == "spair" else "angle"
    if difficulty_key == "viewpoint":
        groups = {str(value): [row for row in rows if row["viewpoint"] == value] for value in (0, 1, 2)}
    else:
        edges = [0, 15, 30, 60, 180] if payload["dataset"] == "scannet" else [0, 30, 60, 90, 120]
        groups = {
            f"[{low},{high})": [row for row in rows if low <= row["angle"] < high]
            for low, high in zip(edges, edges[1:])
        }
    for name, group in groups.items():
        if group:
            by_difficulty[name] = interval(group, first_key, second_key, metric, args)

    output = {
        "protocol": {
            "dataset": payload["dataset"],
            "metric": metric,
            "pairing": "paired over image pairs",
            "resolution": payload["resolution"],
            "evaluation_side": payload["evaluation_side"],
            "num_correspondences": payload["num_correspondences"],
        },
        "model": payload["model"],
        "model_id": payload["model_id"],
        "first": first_key,
        "second": second_key,
        "first_advantage": result,
        "by_difficulty": by_difficulty,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
