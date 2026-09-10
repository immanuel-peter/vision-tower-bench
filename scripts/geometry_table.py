"""Render geometry result JSONs as Markdown tables."""

import argparse
import json
from pathlib import Path

HEADLINE = {"depth": ("d1", "d2", "d3", "rmse"), "normal": ("mean_deg", "d1", "d2", "d3", "rmse")}


def rows(cells: list[dict]) -> list[dict]:
    order = {"tower": 0, "merged": 1, "projected": 2}
    return sorted(cells, key=lambda c: (order[c["stage"]], c["relative_depth"]))


def spread(cell: dict, metric: str) -> str:
    if f"{metric}_std" in cell:
        return f"{cell[metric]:.4f} ± {cell[metric + '_std']:.4f}"
    return f"{cell[metric]:.4f}"


def table(cells: list[dict], task: str) -> str:
    metrics = HEADLINE[task]
    head = ["Stage", "Rel. Depth", "Width", "Grid", "Params", "LR", *metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cell in rows(cells):
        values = [spread(cell, metrics[0])] + [f"{cell[m]:.4f}" for m in metrics[1:]]
        lines.append(
            f"| `{cell['stage']}` | {cell['relative_depth']:.3f} | {cell['token_width']} | "
            f"{cell['grid']}x{cell['grid']} | {cell['trainable_parameters']:,} | "
            f"{cell.get('learning_rate', float('nan')):g} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines)


def scene_table(cells: list[dict], task: str) -> str:
    metric = HEADLINE[task][0]
    head = ["Stage", "Rel. Depth", f"indoors {metric}", f"outdoor {metric}",
            "indoors coverage", "outdoor coverage"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cell in rows(cells):
        scenes = cell["by_scene"]
        lines.append(
            f"| `{cell['stage']}` | {cell['relative_depth']:.3f} | "
            f"{spread(scenes['indoors'], metric)} | {spread(scenes['outdoor'], metric)} | "
            f"{scenes['indoors']['coverage']:.4f} | {scenes['outdoor']['coverage']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--scenes", action="store_true")
    args = ap.parse_args()

    for path in args.files:
        payload = json.loads(path.read_text())
        cells = payload["cells"]
        arm = "capacity-matched" if payload["capacity_matched"] else "unmatched"
        print(f"\n### {cells[0]['model_id']}, {payload['task']}, {arm}\n")
        print(scene_table(cells, payload["task"]) if args.scenes else table(cells, payload["task"]))


if __name__ == "__main__":
    main()
