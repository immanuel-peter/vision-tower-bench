"""Plot measured K3-to-Pro parameter drift without inferring capability changes."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output filename without extension")
    args = parser.parse_args()
    payload = json.loads(args.source.read_text())
    groups = payload["groups"]
    blocks = [groups[f"block_{index:02d}"] for index in range(1, 28)]
    other = ["patch_stem", "spatial_position_embedding", "final_norm", "projector"]
    values = [100 * groups[name]["relative_l2"] for name in other]
    if payload["tensor_count"] != 168 or any(not math.isfinite(row["relative_l2"]) for row in groups.values()):
        raise ValueError("weight drift source is incomplete or nonfinite")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
        axes[0].plot([index / 27 for index in range(1, 28)],
                     [100 * row["relative_l2"] for row in blocks],
                     color="#356d93", marker="o", markersize=3, linewidth=1.6)
        axes[0].set(xlabel="Tower block / 27", ylabel="Relative L2 parameter difference (%)",
                    title="Concatenated parameters per Tower block", ylim=(0, 1.2), xlim=(0, 1.03))
        axes[0].grid(axis="y", alpha=.2)
        bars = axes[1].barh(["Patch stem", "Spatial positions", "Final Tower norm", "Projector"],
                            values, color=["#8aa6b8"] * 3 + ["#b95a3d"])
        axes[1].invert_yaxis()
        axes[1].set(xlabel="Relative L2 parameter difference (%)", xlim=(0, max(values) * 1.2),
                    title="Other parameter groups")
        axes[1].bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=5)
        figure.suptitle("Kimi K3 → K3.1 Pro: measured checkpoint weight differences", fontsize=12)
        figure.text(.02, .015, "||Pro − K3||₂ / ||K3||₂; FP64 sums over each concatenated group. Weight differences do not establish task gains.", fontsize=8)
        figure.tight_layout(rect=(0, .06, 1, .95))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "svg"):
            figure.savefig(args.out.with_suffix(f".{extension}"), dpi=180)
        plt.close(figure)
    args.out.with_suffix(".figure.json").write_text(json.dumps(dict(
        source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), matplotlib_version=matplotlib.__version__,
        method=payload["method"], groups=groups,
        limitation="Parameter drift is descriptive; released checkpoints do not isolate training causes or task gains.",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
