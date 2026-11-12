"""Plot complete matched-attention matrices and actual prediction-based intervals."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("/tmp/vtb-delivery/issues/24")
out = Path("results/diagnostics/kimi-k31-pro-checkpoint")
inputs = [Path("results/recognition/kimi_k31_pro_probe_attention_matched.json"),
          out / "sibling-control/recognition/moonvit_v2_probe_attention_matched.json"]
matrices = [json.loads(path.read_text()) for path in inputs]
book = json.loads((root / "completed-inventory.json").read_text())
intervals = {}
for record in book["completed_paired"]:
    cell = record["cell"]
    if cell["task"] == "recognition" and cell["comparison"] == "kimi_k31_pro-minus-moonvit_v2":
        path = Path("results/bootstrap") / Path(record["source"]).name
        inputs.append(path)
        intervals[cell["first_stage"], cell["first_layer"]] = json.loads(path.read_text())["difference"]
assert len(intervals) == 10
figure, axes = plt.subplots(1, 3, figsize=(12, 4.8))
for payload, name, color in zip(matrices, ("K3.1 Pro", "Fresh K3"), ("#356d93", "#858a8e"), strict=True):
    rows = sorted((row for row in payload["cells"] if row["stage"] == "tower"), key=lambda row: row["layer_index"])
    axes[0].plot([row["relative_depth"] for row in rows], [100 * row["test_accuracy"] for row in rows],
                 marker="o", markersize=3, color=color, label=name)
axes[0].set(xlabel="Relative depth", ylabel="ImageNet-100 accuracy (%)", title="Matched attention, all Tower depths")
axes[0].legend(frameon=False, fontsize=8)
depths = [3, 7, 10, 14, 17, 20, 24, 27]
for index, layer in enumerate(depths):
    ci = intervals["tower", layer]
    axes[1].hlines(index, 100 * ci["lower"], 100 * ci["upper"], color="#356d93")
    axes[1].scatter(100 * ci["point_estimate"], index, color="#356d93", s=18)
axes[1].set(yticks=range(8), yticklabels=[f"L{layer}, {layer/27:.2f}" for layer in depths],
            xlabel="Pro − K3 accuracy (pp)", title="Paired differences by Tower depth")
axes[1].invert_yaxis()
for index, stage in enumerate(("tower", "merged", "projected")):
    ci = intervals[stage, 27]
    axes[2].hlines(index, 100 * ci["lower"], 100 * ci["upper"], color="#356d93")
    axes[2].scatter(100 * ci["point_estimate"], index, color="#356d93", s=18)
axes[2].set(yticks=range(3), yticklabels=("Tower", "Merged", "Projected"), xlabel="Pro − K3 accuracy (pp)", title="Paired final-Stage differences")
axes[2].invert_yaxis()
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.15)
for axis in axes[1:]:
    axis.axvline(0, color="#858a8e", linewidth=.8)
figure.text(.015, .055, "13,000 images; 1,950 paired test IDs; three fitted seeds; 95% per-cell intervals, 10,000 draws, seed0. Same retained fits supply points and intervals.", fontsize=8)
figure.text(.015, .018, "CPU4/8 threads recorded per cell and matched within sibling pairs; fixed GPU1 BF16 batch1 feature producer. No causal training attribution.", fontsize=8)
figure.tight_layout(rect=(0, .13, 1, 1))
for extension in ("svg", "png", "pdf"):
    figure.savefig(out / f"matched-attention-comparison.{extension}", dpi=180)
plt.close(figure)
(out / "matched-attention-comparison.figure.json").write_text(json.dumps(dict(
    inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    intervals=[dict(stage=stage, layer=layer, **ci) for (stage, layer), ci in intervals.items()],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + "\n")
