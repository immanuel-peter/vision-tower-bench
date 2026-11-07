"""Render the isolated Qwen reducer sensitivity without changing primary results."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("qwen38_flash_next", "FlashNext"), ("qwen3_5", "27B")]
STAGES = ["tower", "merged", "projected"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--flash-primary", type=Path, required=True)
    parser.add_argument("--control-primary", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    sources, rows, profiles = {}, [], []

    def load(path):
        sources[os.path.relpath(path, directory)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for (model, name), primary_path in zip(MODELS, [args.flash_primary, args.control_primary]):
        primary = load(primary_path)
        audit = load(directory / f"conditioning-{model}-audit.json")
        diagnostic = load(directory / f"conditioning-{model}-sensitivity.json")
        for stage in STAGES:
            raw_cell = next(c for c in primary["cells"] if c["stage"] == stage)
            diagnostic_cell = next(c for c in diagnostic["cells"] if c["stage"] == stage)
            raw = next(c for c in audit["statistics"] if c["stage"] == stage)
            standardized = next(c for c in diagnostic["conditioning_statistics"] if c["stage"] == stage)
            raw_log = next(c for c in audit["training_losses"] if c["stage"] == stage)
            same_rate = next(c for c in diagnostic["training_losses"]
                             if c["stage"] == stage and c["phase"] == "search"
                             and c["learning_rate"] == raw_log["learning_rate"])
            assert [c["condition"] for c in raw_cell["conditions"]] == [c["condition"] for c in diagnostic_cell["conditions"]]
            profiles.append({"name": name, "stage": stage,
                             "conditions": [c["condition"] for c in raw_cell["conditions"]],
                             "primary_accuracy": [c["accuracy"] for c in raw_cell["conditions"]],
                             "standardized_accuracy": [c["accuracy"] for c in diagnostic_cell["conditions"]]})
            rows.append({
                "model": model, "name": name, "stage": stage,
                "primary_clean_accuracy": raw_cell["conditions"][0]["accuracy"],
                "standardized_clean_accuracy": diagnostic_cell["conditions"][0]["accuracy"],
                "channel_rms_max_to_median": raw["channel_rms_max_to_median"],
                "primary_pc1_share": raw["pc1_share"], "standardized_pc1_share": standardized["pc1_share"],
                "same_rate_seed0_learning_rate": raw_log["learning_rate"],
                "primary_final_training_loss": raw_log["epoch_losses"][-1],
                "standardized_same_rate_final_training_loss": same_rate["epoch_losses"][-1],
                "primary_epoch_losses": raw_log["epoch_losses"],
                "standardized_same_rate_epoch_losses": same_rate["epoch_losses"],
            })
    intervals = {}
    channel_details = load(directory / "channel-details.json")
    assert len(channel_details["rows"]) == 6
    for path in sorted(directory.glob("paired-*.json")):
        data = load(path)
        intervals[path.name] = {"clean": data["clean"], "conditions": {
            key: {k: v for k, v in value.items() if k != "measurements"}
            for key, value in data["conditions"].items()}}
    assert len(intervals) == 4
    output = {"rows": rows, "condition_profiles": profiles, "paired_diagnostic_intervals": intervals, "sources": sources,
              "interpretation": "Readout sensitivity to the reducer; training causes untested. Primary outputs retained."}
    (directory / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    with plt.rc_context({"svg.hashsalt": "qwen-conditioning", "font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        positions = list(range(len(rows)))
        axes[0].bar([p - .18 for p in positions], [100*r["primary_clean_accuracy"] for r in rows],
                    width=.36, label="Primary reducer", color="#888888")
        axes[0].bar([p + .18 for p in positions], [100*r["standardized_clean_accuracy"] for r in rows],
                    width=.36, label="Standardized diagnostic", color="#a55194")
        labels = [r["name"] + "\n" + r["stage"].capitalize() for r in rows]
        axes[0].set(xticks=positions, xticklabels=labels, ylabel="Clean test accuracy (%)", ylim=(0, 100),
                    title="Same prescribed search and three selected seeds")
        axes[0].tick_params(axis="x", labelsize=8)
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].bar([p - .18 for p in positions], [r["primary_final_training_loss"] for r in rows],
                    width=.36, color="#888888")
        axes[1].bar([p + .18 for p in positions], [r["standardized_same_rate_final_training_loss"] for r in rows],
                    width=.36, color="#a55194")
        axes[1].set(xticks=positions, xticklabels=labels, ylabel="Final epoch training cross-entropy", yscale="log",
                    title="Seed 0 at each primary-selected learning rate")
        axes[1].tick_params(axis="x", labelsize=8)
        fig.suptitle("Qwen reducer sensitivity, isolated CPU diagnostic")
        fig.savefig(directory / "conditioning.svg", metadata={"Date": None})
        fig.savefig(directory / "conditioning.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, layout="constrained")
        for axis, profile in zip(axes.flat, profiles):
            positions = list(range(len(profile["conditions"])))
            axis.plot(positions, [100*x for x in profile["primary_accuracy"]], "o-", markersize=3,
                      label="Primary reducer", color="#888888")
            axis.plot(positions, [100*x for x in profile["standardized_accuracy"]], "o-", markersize=3,
                      label="Standardized diagnostic", color="#a55194")
            axis.set(xticks=positions, xticklabels=profile["conditions"], ylim=(0, 100),
                     title=profile["name"] + " " + profile["stage"].capitalize(), ylabel="Test accuracy (%)")
            axis.tick_params(axis="x", labelrotation=65, labelsize=7)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("All 13 conditions, isolated reducer sensitivity")
        fig.savefig(directory / "perturbation-profiles.svg", metadata={"Date": None})
        fig.savefig(directory / "perturbation-profiles.png", dpi=160)
        plt.close(fig)
    table = ["| Checkpoint / Stage | Primary clean | Standardized clean | RMS max/median | PC1 primary | PC1 standardized |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['name']} {r['stage']} | {100*r['primary_clean_accuracy']:.2f}% | "
                     f"{100*r['standardized_clean_accuracy']:.2f}% | {r['channel_rms_max_to_median']:.1f} | "
                     f"{r['primary_pc1_share']:.4f} | {r['standardized_pc1_share']:.4f} |")
    text = """# Qwen perturbation reducer sensitivity

This isolated diagnostic covers both checkpoints, all three Stages, and all 13 conditions.
Every run uses the same CPU hardware, six threads, 1,400 training images, 300 validation
images, and 300 test images. The 11-rate search uses seed 0. The selected rate uses
seeds 0, 1, and 2 for 20 epochs, as in the primary protocol.

The only intervention is the reducer. Clean training tokens alone fit the FP32 channel
mean, sample standard deviation with correction 1 clamped at 1e-6, and seeded PCA.
The frozen reducer and readout then score every condition. No validation or test image
fits the reducer. The exact recovered R2 patch and its SHA256 are recorded in each
sensitivity payload. Primary file hashes match before and after execution.

""" + "\n".join(table) + """

![Clean accuracy and same-rate training loss](conditioning.svg)

![All condition profiles under both reducers](perturbation-profiles.svg)

Standardization recovers the weak FlashNext Tower and merged clean readouts. This
establishes sensitivity to the measurement reducer under the recorded protocol.
Both checkpoints have concentrated Tower variance and large channel RMS ratios,
and their merged PC1 shares are similar. Those statistics alone do not explain why
the primary FlashNext readouts fail. This diagnostic does not establish a training
cause or the same cause as MiMo's readout failure.
The intervention changes channel scales and the retained PCA subspace. This run
does not separate their effects.

The loss comparison holds the seed and learning rate fixed at each primary-selected
rate. The standardized search contains that exact rate, so the plotted loss change
does not depend on selecting a different rate. Full batch and epoch histories, all
search outcomes, and every selected-seed outcome remain in the source payloads.

Four `paired-*.json` files compare standardized Projector against Tower and FlashNext
against fresh 27B at each Stage. Each interval uses its own paired correctness,
the same 300 test IDs, seed means, 10,000 image bootstrap draws, and 95% confidence.
`summary.json` retains these intervals and source hashes. These are diagnostic
comparisons. They do not replace primary measurements or the outstanding full suite.
Primary FlashNext Tower and merged perturbation results do not support intrinsic
robustness claims given their low clean accuracy.

Regenerate the report from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python -m scripts.qwen_conditioning_report \\
  --directory results/diagnostics/qwen38-flash-next-conditioning \\
  --flash-primary results/perturbation/qwen38_flash_next_perturbation.json \\
  --control-primary results/evidence/qwen38-flash-next/controls/qwen3_5_perturbation.json
```

`channel-details.json` retains every training-channel mean, sample standard
deviation, clamped scale, and RMS value, together with exact fit IDs and cache hashes.
`execution-provenance.json` retains the original commands, source hashes,
device settings, start and end times, exit statuses, and output hashes. The hardware
record is `cpu-hardware.json`. The recovered patch is retained in the
[source evidence](../../evidence/qwen38-flash-next/reducer-channel-standardization.patch).
"""
    (directory / "README.md").write_text(text)


if __name__ == "__main__":
    main()
