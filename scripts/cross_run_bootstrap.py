#!/usr/bin/env python3
"""Pair two runs' correspondence or perturbation results and bootstrap the difference.

correspondence_bootstrap.py compares Stages inside one run. This script pairs
two runs by pair or image identity and keeps the paired measurements.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.correspondence_bootstrap import PRIMARY
from vtb.correspondence import paired_bootstrap
from vtb.semantic_bootstrap import paired_image_bootstrap


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("correspondence", "perturbation"))
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--first-name", required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--second-name", required=True)
    ap.add_argument("--stage", default="tower", choices=("tower", "merged", "projected"))
    ap.add_argument("--metric", default=None, help="Correspondence metric; defaults to the dataset's primary metric")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    return ap


def pair_rows(first: list[dict], second: list[dict], key: str) -> list[tuple[dict, dict]]:
    a, b = {row[key]: row for row in first}, {row[key]: row for row in second}
    if len(a) != len(first) or len(b) != len(second):
        raise ValueError(f"duplicate {key}")
    if a.keys() != b.keys() or not a:
        raise ValueError(f"different or empty paired {key} sets")
    return [(a[identity], b[identity]) for identity in sorted(a)]


def stage_cell(payload: dict, stage: str) -> dict:
    cells = [cell for cell in payload["cells"] if cell["stage"] == stage]
    if len(cells) != 1:
        raise ValueError(f"expected one {stage} cell, found {len(cells)}")
    return cells[0]


def correspondence(first: dict, second: dict, names: tuple[str, str], args) -> dict:
    for key in ("dataset", "evaluation_side", "num_correspondences"):
        if first[key] != second[key]:
            raise ValueError(f"correspondence protocol differs: {key}")
    metric = args.metric or PRIMARY[first["dataset"]]
    keys = [f"{cell['stage']}:{cell['layer_index']}" for cell in (stage_cell(first, args.stage), stage_cell(second, args.stage))]
    rows, excluded = [], []
    for a, b in pair_rows(first["measurements"], second["measurements"], "pair_id"):
        values = (a["cells"][keys[0]][metric], b["cells"][keys[1]][metric])
        if not all(math.isfinite(value) for value in values):
            excluded.append(a["pair_id"])
            continue
        rows.append({"pair_id": a["pair_id"], names[0]: values[0], names[1]: values[1]})
    difference = paired_bootstrap(
        torch.tensor([row[names[0]] for row in rows]),
        torch.tensor([row[names[1]] for row in rows]),
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.bootstrap_seed,
    )
    return {
        "metric": metric, "difference": difference, "finite_pairs": len(rows),
        "excluded_nonfinite_pair_ids": excluded, "measurements": rows,
    }


def perturbation(first: dict, second: dict, names: tuple[str, str], args) -> dict:
    if first["images"] != second["images"] or first["protocol"] != second["protocol"]:
        raise ValueError("perturbation probing protocol differs")
    a, b = stage_cell(first, args.stage), stage_cell(second, args.stage)
    if a["measurements"].keys() != b["measurements"].keys():
        raise ValueError("perturbation conditions differ")
    conditions = {}
    for condition in sorted(a["measurements"]):
        rows = []
        for x, y in pair_rows(a["measurements"][condition], b["measurements"][condition], "image_id"):
            rows.append({
                "image_id": x["image_id"], names[0]: x["perturbed"], names[1]: y["perturbed"],
                f"{names[0]}_clean": x["clean"], f"{names[1]}_clean": y["clean"],
            })
        difference = paired_image_bootstrap(
            torch.tensor([row[names[0]] for row in rows]).T,
            torch.tensor([row[names[1]] for row in rows]).T,
            resamples=args.resamples,
            confidence=args.confidence,
            seed=args.bootstrap_seed,
        )
        conditions[condition] = {
            "metric": "perturbed_accuracy", "difference": difference,
            "test_images": len(rows), "measurements": rows,
        }
    return {"conditions": conditions}


def main() -> None:
    args = parser().parse_args()
    first, second = json.loads(args.first.read_text()), json.loads(args.second.read_text())
    names = (args.first_name, args.second_name)
    result = (correspondence if args.kind == "correspondence" else perturbation)(first, second, names, args)
    result["protocol"] = {
        "first_source": str(args.first), "second_source": str(args.second),
        "difference": f"{names[0]}_minus_{names[1]}",
        "bootstrap_seed": args.bootstrap_seed, "resamples": args.resamples, "confidence": args.confidence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote paired {args.kind} intervals to {args.out}")


if __name__ == "__main__":
    main()
