"""Plot complete sibling mean-readout results as descriptive point estimates."""

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
    source = args.results / "diagnostics/kimi-k31-pro-checkpoint"
    records, inputs = [], {}
    for arm in ("raw", "matched"):
        paths = [args.results / "recognition" / f"kimi_k31_pro_probe_mean_{arm}.json",
                 source / "sibling-control/recognition" / f"moonvit_v2_probe_mean_{arm}.json"]
        first, second = [json.loads(path.read_text()) for path in paths]
        if first["protocol"] != second["protocol"] or first["protocol"]["readout"] != "mean":
            raise ValueError("sibling readout protocols differ")
        if first["images"] != 13000 or second["images"] != 13000:
            raise ValueError("complete recognition dataset required")
        a, b = [{(row["stage"], row["layer_index"]): row for row in payload["cells"]}
                for payload in (first, second)]
        if a.keys() != b.keys() or len(a) != 10:
            raise ValueError("complete matching Stage/depth cells required")
        for key in sorted(a):
            x, y = a[key], b[key]
            if x["sample_ids_sha256"] != y["sample_ids_sha256"] or x["split_ids_sha256"] != y["split_ids_sha256"]:
                raise ValueError("sibling actual image/split identities differ")
            if x["seed_ids"] != [0, 1, 2] or y["seed_ids"] != [0, 1, 2]:
                raise ValueError("three retraining seeds required")
            pro, k3 = [sum(row["test_accuracy_by_seed"]) / 3 for row in (x, y)]
            records.append(dict(arm=arm, stage=key[0], layer=key[1],
                                relative_depth=x["relative_depth"], pro_accuracy=pro,
                                k3_accuracy=k3, pro_minus_k3_pp=100 * (pro - k3)))
        inputs.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for axis, arm in zip(axes[:2], ("raw", "matched"), strict=True):
            rows = sorted((row for row in records if row["arm"] == arm and row["stage"] == "tower"), key=lambda row: row["layer"])
            for field, label, color in (("k3_accuracy", "Kimi K3", "#62676b"), ("pro_accuracy", "K3.1 Pro", "#356d93")):
                axis.plot([row["relative_depth"] for row in rows], [100 * row[field] for row in rows],
                          label=label, color=color, marker="o", markersize=3, linewidth=1.5)
            axis.set(title=f"Tower, {arm} capacity", xlabel="Relative depth", ylabel="ImageNet-100 accuracy (%)")
            axis.grid(axis="y", alpha=.2)
            axis.legend(frameon=False, fontsize=8)
        stages = ("tower", "merged", "projected")
        for index, (arm, color) in enumerate((("raw", "#b95a3d"), ("matched", "#356d93"))):
            rows = {row["stage"]: row for row in records if row["arm"] == arm and row["layer"] == 27}
            axes[2].barh([position + (index - .5) * .32 for position in range(3)],
                         [rows[stage]["pro_minus_k3_pp"] for stage in stages], height=.3,
                         color=color, label=arm)
        axes[2].set(yticks=range(3), yticklabels=("Tower", "Merged", "Projected"),
                    xlabel="Pro − K3 accuracy (percentage points)", title="Final Stage difference")
        axes[2].invert_yaxis()
        axes[2].axvline(0, color="#62676b", linewidth=.8)
        axes[2].legend(frameon=False, fontsize=8)
        figure.suptitle("Kimi K3 and K3.1 Pro: complete mean-readout recognition cells", fontsize=12)
        figure.text(.015, .015, "13,000 images; same GPU feature producer and CPU4 fits; three retraining seeds. Point estimates, no paired-image intervals.", fontsize=8)
        figure.tight_layout(rect=(0, .06, 1, .95))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "svg"):
            figure.savefig(args.out.with_suffix(f".{extension}"), dpi=180)
        plt.close(figure)
    args.out.with_suffix(".figure.json").write_text(json.dumps(dict(
        inputs=inputs, records=records,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__,
        limitation="Descriptive mean-readout point estimates; no paired-image uncertainty or causal training attribution.",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
