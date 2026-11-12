"""Render all prescribed Kimi perturbation paired intervals from saved rows."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.results / "diagnostics/kimi-k31-pro-checkpoint/cpu-perturbation/transform-manifest.json"
    source = json.loads(manifest.read_text())
    conditions = source["conditions"]
    labels = ["Clean"]
    for name in conditions[1:]:
        factor, level = name.rsplit("_l", 1)
        index = int(level)
        key = "motion_blur_radius_pixels" if factor == "motion_blur" else factor
        value = source["factors"][key][index]
        labels.append(f"Scale {value:g}×" if factor == "scale" else
                      f"Occlusion {100 * value:g}%" if factor == "occlusion" else f"Blur radius {value:g}px")
    panels = [(f"{stage.title()}: Pro − K3", f"perturbation-kimi_k31_pro-vs-moonvit_v2-{stage}.json")
              for stage in ("tower", "merged", "projected")]
    panels.append(("Pro: Projected − Tower", "perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json"))
    records, hashes = [], {str(manifest): hashlib.sha256(manifest.read_bytes()).hexdigest()}
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(13, 7))
        for panel, (title, filename) in enumerate(panels):
            path = args.results / "bootstrap" / filename
            payload = json.loads(path.read_text())
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            axis = axes[panel]
            for index, condition in enumerate(conditions):
                block = payload["clean"] if condition == "identity" else payload["conditions"][condition]
                interval = block["difference"]
                if block["test_images"] != 300 or interval["resamples"] != 10000 or interval["confidence"] != .95:
                    raise ValueError("paired protocol differs")
                color = "#b95a3d" if panel == 3 else "#356d93"
                axis.hlines(index, 100 * interval["lower"], 100 * interval["upper"], color=color, linewidth=1.5)
                axis.scatter(100 * interval["point_estimate"], index, color=color, s=17, zorder=3)
                records.append(dict(comparison=title, condition=condition, **interval))
            axis.axvline(0, color="#62676b", linewidth=.8)
            axis.set(title=title, yticks=range(13), yticklabels=labels, xlabel="Accuracy difference (pp)")
            axis.grid(axis="x", alpha=.15)
            for boundary in (.5, 4.5, 8.5):
                axis.axhline(boundary, color="#dddddd", linewidth=.6)
        axes[0].invert_yaxis()
        figure.suptitle("Fixed clean-trained attention readouts: all perturbation paired comparisons", fontsize=12)
        figure.text(.015, .025, "2,000 source images; 1,400 train / 300 validation / 300 test; three fitted seeds. Per-cell 95% paired-image intervals, 10,000 draws.", fontsize=8)
        figure.text(.015, .008, "Same GPU1 features and CPU4 fits for siblings. Original train-only PCA retained. These readout results do not isolate intrinsic model robustness.", fontsize=8)
        figure.tight_layout(rect=(0, .06, 1, .95))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "svg"):
            figure.savefig(args.out.with_suffix(f".{extension}"), dpi=180)
        plt.close(figure)
    args.out.with_suffix(".figure.json").write_text(json.dumps(dict(inputs=hashes, records=records,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="Per-cell intervals; fixed readout experiment, without causal training attribution or intrinsic robustness isolation."), indent=2) + "\n")


if __name__ == "__main__":
    main()
