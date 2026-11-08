"""Plot audited matched sibling comparisons using each paired run's own points."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path,
                        default=Path("results/diagnostics/kimi-k31-pro-checkpoint"))
    args = parser.parse_args()
    inventory = load(args.evidence / "completed-inventory.json")
    assert inventory["completed_primary_cells"] == 286
    assert inventory["completed_paired_cells"] == 102
    source_paths = {}
    inputs = {}

    def source(entry):
        name, expected = Path(entry["source"]).name, entry["sha256"]
        if expected not in source_paths:
            candidates = [Path(entry["source"])]
            candidates.extend(Path("results").rglob(name))
            matches = [path for path in candidates if path.is_file() and digest(path) == expected]
            if not matches:
                raise ValueError(f"No content-verified source for {name}")
            source_paths[expected] = matches[0]
            inputs[str(matches[0])] = expected
        return load(source_paths[expected])

    tasks = (("recognition", "Accuracy advantage, percentage points", 100),
             ("depth", "Depth d1 advantage, percentage points", 100),
             ("normal", "Normal error reduction, degrees", 1))
    colors = {"tower": "#29679a", "merged": "#b57a21", "projected": "#8b4a84"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    rows = []
    for ax, (task, label, scale) in zip(axes, tasks, strict=True):
        entries = [entry for entry in inventory["completed_paired"]
                   if entry["cell"]["comparison"] == "kimi_k31_pro-minus-moonvit_v2"
                   and entry["cell"]["task"] == task and entry["cell"].get("arm") == "matched"]
        assert len(entries) == 10
        entries.sort(key=lambda entry: (
            {"tower": 0, "merged": 1, "projected": 2}[entry["cell"]["first_stage"]],
            entry["cell"]["first_layer"]))
        assert [entry["cell"]["first_layer"] for entry in entries[:8]] == [3, 7, 10, 14, 17, 20, 24, 27]
        for entry in entries:
            cell = entry["cell"]
            payload = source(entry)
            interval = payload["difference" if task == "recognition" else "first_advantage"]
            stage = cell["first_stage"]
            x = cell["first_layer"] / 27 if stage == "tower" else {"merged": 1.14, "projected": 1.28}[stage]
            point, low, high = [scale * interval[name] for name in ("point_estimate", "lower", "upper")]
            assert low <= point <= high
            ax.errorbar(x, point, yerr=[[point - low], [high - point]], fmt="o",
                        color=colors[stage], capsize=3, markersize=4)
            rows.append(dict(task=task, stage=stage, layer=cell["first_layer"],
                             relative_depth=cell["first_layer"] / 27, point=point,
                             lower=low, upper=high, units=label, source_sha256=entry["sha256"],
                             source=str(source_paths[entry["sha256"]])))
        ax.axhline(0, color="0.45", linewidth=.8)
        ax.axvline(1.07, color="0.75", linewidth=.7, linestyle=":")
        ax.set_xticks([3 / 27, .5, 1, 1.14, 1.28], [".11", ".50", "1.0", "Merged", "Projected"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Tower Relative Depth / final Stages")
        ax.set_ylabel(label)
        ax.set_title({"recognition": "Recognition", "depth": "DIODE depth", "normal": "DIODE normals"}[task])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("K3.1 Pro versus fresh K3, matched readouts and paired 95% intervals", fontsize=12)
    stem = args.evidence / "matched-sibling-relative-depth-intervals"
    for extension in ("svg", "png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=180)
    plt.close(fig)
    manifest = dict(
        point_source="Actual paired payload points and intervals; no matrix point substitution",
        normal_direction="K3 error minus Pro error; positive favors Pro",
        statistical_scope="Paired test-image resampling, with three-seed means; individual 95% intervals, no multiplicity adjustment",
        hardware_scope="Both arms of each comparison share physical device and settings; device provenance is retained per source run",
        limitation="Production official vLLM TORCH_SDPA BF16 parity failed; accepted conversion parity uses an independent mathematical reference",
        inputs_sha256=inputs, inventory_sha256=digest(args.evidence / "completed-inventory.json"),
        rows=rows,
    )
    stem.with_suffix(".figure.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered {len(rows)} actual sibling intervals to {stem}")


if __name__ == "__main__":
    main()
