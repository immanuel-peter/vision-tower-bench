#!/usr/bin/env python3
"""Paired image-bootstrap intervals for the FSC-147 count readouts and CountQA transfer.

FSC-147 (``fsc``) reads every ``<model>_count_<free|exemplar>_<raw|matched>.json`` under
``--results`` (complete payloads; otherwise finished per-cell JSON + TSV pairs, marked partial)
and writes, per model:

- Stage pairs inside one arm and capacity: final Tower vs merged, merged vs projected,
  final Tower vs projected, on identical test IDs.
- Exemplar-free vs exemplar at the final Tower, same capacity, on the exemplar arm's test IDs.
- Raw vs matched at the same Stage and arm.

CountQA (``countqa``) reads ``scripts/countqa_eval.py`` outputs and writes their Stage pairs,
plus free vs exemplar at the Tower when both arms of one model and variant are given.

Outputs: ``<out>/<dataset>/<model>_count_bootstrap.json`` (protocol, sources with sha256,
comparisons with overall and per-bin MAE and RMSE intervals) and ``<out>/<dataset>/count_intervals.tsv``.
Statistics: see ``vtb/count_bootstrap.py``.

    uv run python -m scripts.count_bootstrap fsc --results $VTB_WORK/results/fsc147 --out $VTB_WORK/analysis
    uv run python -m scripts.count_bootstrap countqa results/counting/countqa/*_count_*.json --out $VTB_WORK/analysis
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from vtb import count_bootstrap as cb

TSV_COLUMNS = ("dataset", "model", "comparison", "family", "first", "second", "status", "metric", "bin",
               "images", "small_sample", "first_value", "second_value", "point_estimate", "lower", "upper",
               "favours")


def parser() -> argparse.ArgumentParser:
    work = Path(os.environ.get("VTB_WORK", Path.home() / "vtb-work" / "counting-v3"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="dataset", required=True)
    fsc = sub.add_parser("fsc", help="FSC-147 count results")
    fsc.add_argument("--results", type=Path, default=work / "results" / "fsc147")
    fsc.add_argument("--models", nargs="+", default=None)
    fsc.add_argument("--complete-only", action="store_true",
                     help="ignore per-cell files of incomplete payloads")
    qa = sub.add_parser("countqa", help="CountQA transfer results from scripts/countqa_eval.py")
    qa.add_argument("files", nargs="+", type=Path)
    for p in (fsc, qa):
        p.add_argument("--out", type=Path, default=work / "analysis")
        p.add_argument("--relative-to", type=Path, default=None,
                       help="record source paths relative to this directory when they lie under it")
        p.add_argument("--resamples", type=int, default=cb.RESAMPLES)
        p.add_argument("--confidence", type=float, default=cb.CONFIDENCE)
        p.add_argument("--bootstrap-seed", type=int, default=cb.SEED)
        p.add_argument("--label", default=None, help="e.g. 'pilot'; stored in every output")
        p.add_argument("--no-image-ids", action="store_true", help="omit the per-comparison image ID lists")
    return ap


def relative(path: str, root: Path | None) -> str:
    if root is None:
        return path
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return path


def comparisons_for(results: list[cb.Result], kwargs: dict) -> dict[str, dict]:
    by_key = {(r.arm, r.capacity): r for r in results}
    out: dict[str, dict] = {}
    for result in results:
        out |= cb.stage_comparisons(result, **kwargs)
    for capacity in ("raw", "matched"):
        if ("free", capacity) in by_key and ("exemplar", capacity) in by_key:
            out |= cb.arm_comparison(by_key["free", capacity], by_key["exemplar", capacity], **kwargs)
    for arm in ("free", "exemplar"):
        if (arm, "raw") in by_key and (arm, "matched") in by_key:
            out |= cb.capacity_comparisons(by_key[arm, "raw"], by_key[arm, "matched"], **kwargs)
    return out


def summary_rows(dataset: str, model: str, comparisons: dict[str, dict]) -> list[list]:
    rows = []
    for name, entry in comparisons.items():
        status = ",".join(sorted({entry["first"]["status"], entry["second"]["status"]}))
        blocks = [("all", entry)] + [(b, entry["by_bin"][b]) for b in cb.BINS]
        for bin_name, block in blocks:
            if "mae" not in block:
                rows.append([dataset, model, name, entry["family"], entry["first"]["cell"], entry["second"]["cell"],
                             status, "", bin_name, block["images"], block.get("small_sample", False),
                             "", "", "", "", "", ""])
                continue
            for metric, value_key in (("mae", "mae"), ("rmse", "rmse"), ("rmse_pooled", "rmse_pooled")):
                interval = block["rmse"]["pooled"] if metric == "rmse_pooled" else block[metric]
                favours = cb.resolved(interval)
                rows.append([
                    dataset, model, name, entry["family"], entry["first"]["cell"], entry["second"]["cell"], status,
                    metric, bin_name, block["images"], block.get("small_sample", False),
                    f"{block['first_summary'][value_key]:.4f}", f"{block['second_summary'][value_key]:.4f}",
                    f"{interval['point_estimate']:.4f}", f"{interval['lower']:.4f}", f"{interval['upper']:.4f}",
                    favours,
                ])
    return rows


def write(dataset: str, grouped: dict[str, list[cb.Result]], args) -> list[Path]:
    kwargs = {"resamples": args.resamples, "confidence": args.confidence, "seed": args.bootstrap_seed}
    out_dir = args.out / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    written, rows = [], []
    for model, results in sorted(grouped.items()):
        comparisons = comparisons_for(results, kwargs)
        if not comparisons:
            print(f"{dataset} {model}: no pairable cells yet")
            continue
        if args.no_image_ids:
            for entry in comparisons.values():
                entry.pop("image_ids", None)
        sources = {relative(path, args.relative_to): sha for r in results for path, sha in sorted(r.sources.items())}
        payload = {
            "dataset": dataset, "model": model, "label": args.label,
            "partial": any(r.status != "complete" for r in results),
            "protocol": cb.protocol(args.resamples, args.confidence, args.bootstrap_seed),
            "results": [{"arm": r.arm, "capacity": r.capacity, "status": r.status, "cells": sorted(r.cells)}
                        for r in results],
            "sources": sources,
            "comparisons": comparisons,
        }
        path = out_dir / f"{model}_count_bootstrap.json"
        path.write_text(json.dumps(payload, indent=1) + "\n")
        written.append(path)
        rows += summary_rows(dataset, model, comparisons)
        for name, entry in comparisons.items():
            mae, rmse = entry["mae"], entry["rmse"]
            print(f"{dataset} {model:<14} {name:<44} n={entry['images']:<5} "
                  f"dMAE {mae['point_estimate']:+8.3f} [{mae['lower']:+8.3f}, {mae['upper']:+8.3f}]  "
                  f"dRMSE {rmse['point_estimate']:+8.3f} [{rmse['lower']:+8.3f}, {rmse['upper']:+8.3f}]")
    if rows:
        tsv = out_dir / "count_intervals.tsv"
        lines = ["\t".join(TSV_COLUMNS)] + ["\t".join(str(v) for v in row) for row in rows]
        tsv.write_text("\n".join(lines) + "\n")
        written.append(tsv)
    return written


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    grouped: dict[str, list[cb.Result]] = defaultdict(list)
    if args.dataset == "fsc":
        for result in cb.discover(args.results, allow_partial=not args.complete_only):
            if args.models is None or result.model in args.models:
                grouped[result.model].append(result)
        written = write("fsc147", grouped, args)
    else:
        for path in args.files:
            result = cb.load_countqa(path)
            grouped[result.model].append(result)
        written = write("countqa", grouped, args)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
