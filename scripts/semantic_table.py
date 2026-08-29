"""Render the semantic result JSONs as the markdown tables the writeup carries.

The geometry twin of this script is scripts/geometry_table.py. Both take the JSON a lane
script wrote and print one table per file, so the writeup never retypes a number.
"""

import argparse
import json
from pathlib import Path


def rows(cells: list[dict]) -> list[dict]:
    """Tower depth points first, then the Stages that exist only at the deepest point."""
    order = {"tower": 0, "merged": 1, "projected": 2}
    return sorted(cells, key=lambda c: (order[c["stage"]], c["relative_depth"]))


def table(cells: list[dict]) -> str:
    head = ["Stage", "Rel. Depth", "Width", "Params", "LR", "top-1", "val"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cell in rows(cells):
        lines.append(
            f"| `{cell['stage']}` | {cell['relative_depth']:.3f} | {cell['token_width']} | "
            f"{cell['trainable_parameters']:,} | {cell['learning_rate']:g} | "
            f"{cell['test_accuracy']:.4f} ± {cell['test_std']:.4f} | "
            f"{cell['val_accuracy']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    for path in args.files:
        payload = json.loads(path.read_text())
        cells = payload["cells"]
        readout, _, arm = path.stem.partition("_probe_")[2].partition("_")
        arm = "capacity-matched" if arm == "matched" else "unmatched"
        print(f"\n### {cells[0]['model_id']}, {readout} readout, {arm}\n")
        print(table(cells))


if __name__ == "__main__":
    main()
