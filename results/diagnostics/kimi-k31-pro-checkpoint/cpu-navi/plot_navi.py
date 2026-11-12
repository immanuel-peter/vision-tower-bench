"""Render audited NAVI Stage scores and paired differences."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path("results/diagnostics/kimi-k31-pro-checkpoint")
pro = Path("results/correspondence/kimi_k31_pro_correspondence_navi.json")
k3 = output / "sibling-control/correspondence/moonvit_v2_correspondence_navi.json"
inputs = [pro, k3]
values = [{row["stage"]: row["recall_2cm"] for row in json.loads(path.read_text())["cells"]} for path in inputs]
stages = ("tower", "merged", "projected")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for index, (data, label, color) in enumerate(zip(values, ("K3.1 Pro", "Fresh K3"), ("#356d93", "#858a8e"), strict=True)):
    axes[0].bar([position + (index - .5) * .34 for position in range(3)], [100 * data[stage] for stage in stages],
                width=.32, color=color, label=label)
axes[0].set(xticks=range(3), xticklabels=("Tower", "Merged", "Projected"), ylabel="Correspondence recall within 2cm (%)", title="NAVI: all 555 selected pairs")
axes[0].legend(frameon=False)
records = []
for index, stage in enumerate(stages):
    path = Path("results/bootstrap") / f"correspondence-kimi_k31_pro-vs-moonvit_v2-navi-{stage}.json"
    inputs.append(path)
    interval = json.loads(path.read_text())["difference"]
    axes[1].hlines(index, 100 * interval["lower"], 100 * interval["upper"], color="#356d93")
    axes[1].scatter(100 * interval["point_estimate"], index, color="#356d93")
    records.append(dict(stage=stage, **interval))
axes[1].axvline(0, color="#858a8e", linewidth=.8)
axes[1].set(yticks=range(3), yticklabels=("Tower", "Merged", "Projected"), xlabel="Pro − K3 recall (percentage points)", title="95% paired bootstrap intervals")
axes[1].invert_yaxis()
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
fig.text(.015, .015, "BF16 batch1 CPU4 vision inference; native grids, evaluation64, 1,000 matches; 10,000 paired resamples, seed0. Per-cell intervals.", fontsize=8)
fig.tight_layout(rect=(0, .06, 1, 1))
for extension in ("svg", "png", "pdf"):
    fig.savefig(output / f"navi-stage-comparison.{extension}", dpi=180)
plt.close(fig)
(output / "navi-stage-comparison.figure.json").write_text(json.dumps(dict(records=records,
    inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitation="Frozen-feature matching on the same CPU4 hardware; no causal training attribution or production BF16 agreement claim."), indent=2) + "\n")
