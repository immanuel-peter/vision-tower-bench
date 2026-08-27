"""Compare geometry results against seed spread as specified by ADR-0013."""

import argparse
import json
from pathlib import Path

# Selection metric and direction for each task.
HEADLINE = {"depth": ("d1", True), "normal": ("mean_deg", False)}


def load(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["arm"] = "matched" if payload["capacity_matched"] else "unmatched"
        payload["model"] = payload["cells"][0]["model_id"]
        runs.append(payload)
    return sorted(runs, key=lambda r: (r["model"], r["task"], r["arm"]))


def label(run: dict) -> str:
    return f"{run['model']:24s} {run['task']:6s} {run['arm']:9s}"


def towers(run: dict) -> list[dict]:
    return sorted((c for c in run["cells"] if c["stage"] == "tower"), key=lambda c: c["relative_depth"])


def deepest(run: dict, stage: str) -> dict | None:
    cells = [c for c in run["cells"] if c["stage"] == stage]
    return max(cells, key=lambda c: c["relative_depth"]) if cells else None


def gap(left: dict, right: dict, metric: str) -> tuple[float, float]:
    """Signed change from left to right, and its size in pooled seed deviations."""
    spread = ((left[f"{metric}_std"] ** 2 + right[f"{metric}_std"] ** 2) / 2) ** 0.5
    change = right[metric] - left[metric]
    return change, abs(change) / spread if spread else float("inf")


def peaks(runs: list[dict]) -> None:
    print("== Tower peak, and the fall from it to the final layer ==")
    for run in runs:
        metric, higher = HEADLINE[run["task"]]
        rows = towers(run)
        best = (max if higher else min)(rows, key=lambda c: c[metric])
        last = rows[-1]
        runner = (max if higher else min)((c for c in rows if c is not best), key=lambda c: c[metric])
        margin, margin_sd = gap(runner, best, metric)
        drop, drop_sd = gap(best, last, metric)
        print(
            f"{label(run)} peak {best['relative_depth']:.3f} "
            f"{metric} {best[metric]:.4f} +- {best[f'{metric}_std']:.4f} lr {best['learning_rate']:g}"
        )
        print(
            f"{'':41s} over runner-up {runner['relative_depth']:.3f} by {abs(margin):.4f} "
            f"({margin_sd:.1f} sd), falls to {last[metric]:.4f} at {last['relative_depth']:.3f} "
            f"by {abs(drop):.4f} ({drop_sd:.1f} sd)"
        )


def stages(runs: list[dict]) -> None:
    print("\n== Stage steps at the deepest point ==")
    for run in runs:
        metric = HEADLINE[run["task"]][0]
        tower, merged, projected = (deepest(run, s) for s in ("tower", "merged", "projected"))
        if not merged or not projected:
            continue
        for name, cell in (("tower", tower), ("merged", merged), ("projected", projected)):
            print(
                f"{label(run)} {name:>9} {metric} {cell[metric]:.4f} +- {cell[f'{metric}_std']:.4f} "
                f"lr {cell['learning_rate']:g} width {cell['token_width']} "
                f"params {cell['trainable_parameters']:,}"
            )
        for name, left, right in (
            ("tower->merged", tower, merged),
            ("merged->projected", merged, projected),
            ("tower->projected", tower, projected),
        ):
            change, deviations = gap(left, right, metric)
            print(f"{label(run)} {name:>17} {change:+.4f} ({deviations:.1f} sd)")


def rates(runs: list[dict]) -> None:
    print("\n== Selected learning rate, by patch grid ==")
    counts: dict[tuple[int, str], dict[str, int]] = {}
    for run in runs:
        for cell in run["cells"]:
            key = (cell["grid"], run["task"])
            rate = f"{cell['learning_rate']:g}"
            counts.setdefault(key, {})[rate] = counts.setdefault(key, {}).get(rate, 0) + 1
    for (grid, task), tally in sorted(counts.items()):
        chosen = ", ".join(f"{rate} x{n}" for rate, n in sorted(tally.items(), key=lambda kv: float(kv[0])))
        print(f"{grid}x{grid} {task:6s} {chosen}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    runs = load(args.files)
    peaks(runs)
    stages(runs)
    rates(runs)


if __name__ == "__main__":
    main()
