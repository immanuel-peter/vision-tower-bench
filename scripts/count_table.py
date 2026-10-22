#!/usr/bin/env python3
"""Per-model, Stage, arm and capacity count table with per-bin MAE/RMSE and sample counts.

Every number is recomputed from the contract prediction TSVs (the same rows the bootstrap
resamples): MAE is the mean over seeds of each seed's MAE, RMSE the mean over seeds of each
seed's RMSE (as in the cell JSON), pooled RMSE the square root of the mean squared error over all
seed x image rows. Bins use the true count (le50, 50to200, gt200); bins with fewer than 30
images are flagged. The exemplar arm's one-exemplar FSC validation result is a transfer
diagnostic (cell JSON ``one_exemplar_val``), never used for selection.

    uv run python -m scripts.count_table --results $VTB_WORK/results/fsc147 \
        [--countqa results/counting/countqa/*_count_*.json] [--out DIR] [--label pilot]
"""

import argparse
import os
from pathlib import Path

import numpy as np

from vtb import count_bootstrap as cb

TSV_COLUMNS = ("dataset", "model", "arm", "capacity", "stage", "layer", "test_ids", "bin", "images",
               "small_sample", "mae", "mae_std", "rmse", "rmse_std", "rmse_pooled", "bias", "exact_match",
               "learning_rate", "status")
ARM_ORDER = {"free": 0, "exemplar": 1}
CAPACITY_ORDER = {"raw": 0, "matched": 1}


def _row(result: cb.Result, cell: cb.Cell, test_ids: str) -> list[dict]:
    rows = []
    blocks = [("all", cb.cell_summary(cell) | {"small_sample": len(cell.image_ids) < cb.SMALL_BIN})]
    blocks += list(cb.by_bin_summary(cell).items())
    exact = None
    for bin_name, summary in blocks:
        if result.dataset == "countqa" and summary.get("images"):
            ids = cell.image_ids if bin_name == "all" else [i for i, b in zip(cell.image_ids, cell.bins) if b == bin_name]
            sub = cell.subset(ids)
            exact = float((np.floor(sub.predicted.numpy() + 0.5) == sub.true[None, :]).mean())
        rows.append({
            "dataset": result.dataset, "model": result.model, "arm": result.arm, "capacity": result.capacity,
            "stage": cell.stage, "layer": cell.layer, "test_ids": test_ids, "bin": bin_name,
            "images": summary.get("images", 0), "small_sample": summary["small_sample"],
            "mae": summary.get("mae"), "mae_std": summary.get("mae_std"), "rmse": summary.get("rmse"),
            "rmse_std": summary.get("rmse_std"), "rmse_pooled": summary.get("rmse_pooled"),
            "bias": summary.get("bias"), "exact_match": exact if summary.get("images") else None,
            "learning_rate": cell.meta.get("learning_rate"), "status": result.status,
        })
    return rows


def _order(result: cb.Result) -> tuple:
    return result.model, ARM_ORDER.get(result.arm, 9), CAPACITY_ORDER.get(result.capacity, 9)


def fsc_rows(results: list[cb.Result]) -> list[dict]:
    """One row per cell and bin; the free Tower is also given on the exemplar arm's test IDs."""
    rows = []
    exemplar = {(r.model, r.capacity): r for r in results if r.arm == "exemplar"}
    for result in sorted(results, key=_order):
        cells = sorted(result.cells.values(), key=lambda c: (cb.STAGES.index(c.stage), c.layer))
        for cell in cells:
            label = "eligible" if result.arm == "exemplar" else "all"
            rows += _row(result, cell, label)
            partner = exemplar.get((result.model, result.capacity))
            final = result.final("tower")
            if result.arm == "free" and partner is not None and cell is final and partner.final("tower"):
                rows += _row(result, cell.subset(partner.final("tower").image_ids), "eligible")
    return rows


def one_exemplar_rows(results: list[cb.Result]) -> list[dict]:
    """Exemplar-arm validation with only release box 0 next to the same heads' three-box validation."""
    rows = []
    for result in sorted(results, key=_order):
        if result.arm != "exemplar":
            continue
        for key, cell_json in sorted(result.cell_json.items()):
            one = cell_json.get("one_exemplar_val")
            if not one:
                continue
            val = result.val.get(key)
            rows.append({
                "model": result.model, "capacity": result.capacity, "stage": cell_json["stage"],
                "layer": cell_json["layer_index"], "val_images": one.get("images"),
                "three_box_val_mae": cb.cell_summary(val)["mae"] if val is not None else None,
                "three_box_val_rmse": cb.cell_summary(val)["rmse"] if val is not None else None,
                "one_exemplar_val_mae": one.get("mae"), "one_exemplar_val_mae_std": one.get("mae_std"),
                "one_exemplar_val_rmse": one.get("rmse"), "status": result.status,
            })
    return rows


def fmt(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def headline_markdown(rows: list[dict]) -> str:
    head = ["Model", "Arm", "Capacity", "Stage", "L", "Test IDs", "n", "MAE ± sd", "RMSE ± sd", "pooled RMSE",
            "bias", "LR", "status"]
    has_exact = any(r["exact_match"] is not None for r in rows)
    if has_exact:
        head.insert(-2, "exact")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if r["bin"] != "all":
            continue
        values = [r["model"], r["arm"], r["capacity"], f"`{r['stage']}`", str(r["layer"]), r["test_ids"],
                  str(r["images"]), f"{fmt(r['mae'])} ± {fmt(r['mae_std'])}",
                  f"{fmt(r['rmse'])} ± {fmt(r['rmse_std'])}", fmt(r["rmse_pooled"]), fmt(r["bias"]),
                  "n/a" if r["learning_rate"] is None else f"{r['learning_rate']:g}", r["status"]]
        if has_exact:
            values.insert(-2, fmt(r["exact_match"], 3))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def bin_markdown(rows: list[dict]) -> str:
    head = ["Model", "Arm", "Capacity", "Stage", "Test IDs"]
    for name in cb.BINS:
        head += [f"{name} n", f"{name} MAE", f"{name} RMSE"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    grouped: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], r["arm"], r["capacity"], r["stage"], r["layer"], r["test_ids"])
        grouped.setdefault(key, {})[r["bin"]] = r
    for key, bins in grouped.items():
        values = [key[0], key[1], key[2], f"`{key[3]}`", key[5]]
        for name in cb.BINS:
            r = bins[name]
            flag = "†" if r["small_sample"] else ""
            values += [f"{r['images']}{flag}", fmt(r["mae"]), fmt(r["rmse"])]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def diagnostic_markdown(rows: list[dict]) -> str:
    head = ["Model", "Capacity", "Stage", "val n", "3-box val MAE", "1-box val MAE ± sd", "3-box val RMSE",
            "1-box val RMSE"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"], r["capacity"], f"`{r['stage']}`", str(r["val_images"]), fmt(r["three_box_val_mae"]),
            f"{fmt(r['one_exemplar_val_mae'])} ± {fmt(r['one_exemplar_val_mae_std'])}",
            fmt(r["three_box_val_rmse"]), fmt(r["one_exemplar_val_rmse"])]) + " |")
    return "\n".join(lines)


def markdown(fsc: list[dict], diagnostics: list[dict], countqa: list[dict], label: str | None) -> str:
    parts = []
    if label:
        parts.append(f"> **{label.upper()}**: {label} numbers, not final results.\n")
    if fsc:
        parts += ["## FSC-147 test (official split)\n",
                  "MAE and RMSE: mean ± sd over seeds of each seed's value. Pooled RMSE: sqrt of the mean "
                  "squared error over all seed x image rows. Test IDs `eligible`: the exemplar arm's test IDs "
                  "(all three release boxes valid after the crop); the free Tower is repeated on them for the "
                  "arm comparison. Partial results use finished cells of an incomplete run.\n",
                  headline_markdown(fsc), "",
                  "### Per count bin (true count; † marks n < 30)\n", bin_markdown(fsc), ""]
    if diagnostics:
        parts += ["### Diagnostic: one-exemplar FSC-147 validation\n",
                  "Transfer diagnostic only, never used for selection: the three-box exemplar head read with "
                  "release box 0 alone on FSC-147 validation (`one_exemplar_val`), beside the same heads' "
                  "three-box validation error from the cell's `.val.tsv`.\n",
                  diagnostic_markdown(diagnostics), ""]
    if countqa:
        parts += ["## CountQA transfer (curated subset; frozen FSC-147 heads)\n",
                  "Exemplar-based frozen-feature counting on a curated CountQA subset; not MLLM question "
                  "answering and not comparable to MLLM CountQA scores. Exact match: floor(pred + 0.5) == answer.\n",
                  headline_markdown(countqa), "",
                  "### Per count bin (true answer; † marks n < 30)\n", bin_markdown(countqa), ""]
    return "\n".join(parts) + "\n"


def tsv(rows: list[dict]) -> str:
    def cell(value):
        if value is None:
            return ""
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    return "\n".join(["\t".join(TSV_COLUMNS)] + ["\t".join(cell(r[c]) for c in TSV_COLUMNS) for r in rows]) + "\n"


def parser() -> argparse.ArgumentParser:
    work = Path(os.environ.get("VTB_WORK", Path.home() / "vtb-work" / "counting-v3"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=work / "results" / "fsc147")
    ap.add_argument("--countqa", type=Path, nargs="*", default=[])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--complete-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write count_table.md and count_table.tsv here")
    ap.add_argument("--label", default=None, help="e.g. 'pilot'")
    return ap


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    results = [r for r in cb.discover(args.results, allow_partial=not args.complete_only)
               if args.models is None or r.model in args.models]
    qa = [cb.load_countqa(p) for p in args.countqa]
    qa = [r for r in qa if args.models is None or r.model in args.models]
    fsc, diagnostics = fsc_rows(results), one_exemplar_rows(results)
    countqa = fsc_rows(qa)
    text = markdown(fsc, diagnostics, countqa, args.label)
    if args.out is None:
        print(text)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "count_table.md").write_text(text)
    (args.out / "count_table.tsv").write_text(tsv(fsc + countqa))
    print(f"wrote {args.out / 'count_table.md'} and {args.out / 'count_table.tsv'}")


if __name__ == "__main__":
    main()
