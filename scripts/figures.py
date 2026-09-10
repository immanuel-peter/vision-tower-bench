#!/usr/bin/env python3
"""Render the blog figures from committed result JSON. No GPU, no feature cache."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
TOWERS: tuple[str, ...] = (
    "dinov2", "siglip2", "muse_glimmer", "kimi_k26", "qwen3_5", "moonvit_v2",
)
COLOUR: dict[str, tuple] = dict(zip(TOWERS, plt.get_cmap("tab10").colors))

# Report names. Plots and tables must agree, and the two Kimi Towers are named for
# the model that ships them rather than for the encoder family.
DISPLAY: dict[str, str] = {
    "dinov2": "DINOv2", "siglip2": "SigLIP2", "muse_glimmer": "Muse Glimmer",
    "kimi_k26": "Kimi K2.6", "qwen3_5": "Qwen3.8", "moonvit_v2": "Kimi K3",
}


def tower_name(label: str) -> str:
    return next(t for t in TOWERS if label.startswith(t))

# Panel order and axis label for every Projector comparison, keyed by the slug that sits
# between "geometry-"/"correspondence-" and the model name in results/bootstrap.
PANELS = {
    "depth-matched": "DIODE depth, matched (d1)",
    "depth-raw": "DIODE depth, raw (d1)",
    "normal-matched": "DIODE normals, matched (degrees)",
    "normal-raw": "DIODE normals, raw (degrees)",
    "depth-kitti-matched": "KITTI depth, matched (d1)",
    "navi": "NAVI correspondence (recall@2cm)",
    "scannet": "ScanNet correspondence (recall@10px)",
    "spair": "SPair-71k correspondence (PCK@0.1)",
}


def projector_intervals() -> dict[str, list[tuple[str, float, float, float]]]:
    panels = defaultdict(list)
    for path in sorted(RESULTS.glob("bootstrap/*-projected-vs-tower.json")):
        stem = path.stem.removesuffix("-projected-vs-tower")
        rest = stem.split("-", 1)[1]
        model = next(t for t in TOWERS if rest.endswith(t))
        slug = rest.removesuffix(f"-{model}").rstrip("-")
        advantage = json.loads(path.read_text())["first_advantage"]
        panels[slug].append(
            (model, advantage["point_estimate"], advantage["lower"], advantage["upper"])
        )
    return panels


def forest(out: Path) -> None:
    panels = projector_intervals()
    keys = [k for k in PANELS if k in panels]
    figure, axes = plt.subplots(
        len(keys), 1, figsize=(7, 1.1 * sum(len(panels[k]) for k in keys) + 1.6),
        gridspec_kw={"height_ratios": [len(panels[k]) for k in keys]},
    )
    for axis, key in zip(axes, keys):
        rows = sorted(panels[key], key=lambda r: r[1])
        for y, (model, point, low, high) in enumerate(rows):
            resolved = low > 0 or high < 0
            axis.plot([low, high], [y, y], color=COLOUR[model], linewidth=2)
            axis.plot(
                [point], [y], "o", color=COLOUR[model], markersize=6,
                markerfacecolor=COLOUR[model] if resolved else "white",
            )
        axis.axvline(0, color="0.3", linewidth=1, linestyle="--")
        axis.set_yticks(range(len(rows)), [DISPLAY[r[0]] for r in rows], fontsize=8)
        axis.set_ylim(-0.6, len(rows) - 0.4)
        axis.set_title(PANELS[key], fontsize=9, loc="left")
        axis.tick_params(labelsize=8)
    figure.suptitle(
        "Projector advantage over the final Tower layer\n"
        "95% paired bootstrap intervals; hollow markers cross zero",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def label_budget(out: Path, readout: str = "attention") -> None:
    series = defaultdict(dict)
    for path in sorted(RESULTS.glob(f"label-budget/*_probe_{readout}_matched_f*.json")):
        model = path.stem.split("_probe_")[0]
        payload = json.loads(path.read_text())
        series[model][payload["label_fraction"]] = payload["cells"][0]["test_accuracy"]

    figure, axis = plt.subplots(figsize=(6, 4))
    for model in TOWERS:
        points = sorted(series[model].items())
        axis.plot(
            [f for f, _ in points], [a for _, a in points],
            marker="o", markersize=4, color=COLOUR[model], label=DISPLAY[model],
        )
    axis.set_xscale("log")
    axis.set_xticks([0.01, 0.05, 0.20, 1.0], ["1%", "5%", "20%", "100%"])
    axis.set_xlabel("labelled fraction of the training split (log scale)")
    axis.set_ylabel(f"recognition top-1 ({readout}, matched)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def relative_depth(out: Path, readout: str = "attention", arm: str = "matched") -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    for model in TOWERS:
        path = RESULTS / f"{model}_probe_{readout}_{arm}.json"
        cells = [c for c in json.loads(path.read_text())["cells"] if c["stage"] == "tower"]
        cells.sort(key=lambda c: c["relative_depth"])
        axis.plot(
            [c["relative_depth"] for c in cells], [c["test_accuracy"] for c in cells],
            marker="o", markersize=3, color=COLOUR[model], label=DISPLAY[model],
        )
    axis.set_xlabel("Relative Depth")
    axis.set_ylabel(f"top-1 accuracy, {readout} readout, {arm}")
    axis.set_title("Semantic decodability rises toward the last layers", fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def occlusion(out: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    for path in sorted(RESULTS.glob("perturbation/*_perturbation.json")):
        payload = json.loads(path.read_text())
        model = tower_name(payload["model"])
        cell = next(c for c in payload["cells"] if c["stage"] == "tower")
        points = sorted((c["value"], c["accuracy"]) for c in cell["conditions"])
        axis.plot(
            [v for v, _ in points], [a for _, a in points],
            marker="o", markersize=4, color=COLOUR[model], label=DISPLAY[model],
        )
    axis.set_xlabel("occluded fraction of the image")
    axis.set_ylabel("top-1 accuracy")
    axis.set_title("Occlusion splits the roster into two robustness groups", fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def capability_profile(out: Path) -> None:
    axes_names = ["semantic 1%", "semantic 100%", "NAVI", "ScanNet", "SPair", "occlusion 50%"]
    scores: dict[str, list[float]] = {m: [] for m in TOWERS}

    for readout_key in ("f001", "f10"):
        for model in TOWERS:
            path = RESULTS / f"label-budget/{model}_probe_attention_matched_{readout_key}.json"
            scores[model].append(json.loads(path.read_text())["cells"][0]["test_accuracy"])

    for dataset, metric in (("navi", "recall_2cm"), ("scannet", "recall_10px"), ("spair", "pck_0.1")):
        for model in TOWERS:
            payload = json.loads((RESULTS / f"correspondence/{model}_correspondence_{dataset}.json").read_text())
            tower = next(c for c in payload["cells"] if c["stage"] == "tower")
            scores[model].append(tower[metric])

    for model in TOWERS:
        payload = json.loads((RESULTS / f"perturbation/{model}_perturbation.json").read_text())
        cell = next(c for c in payload["cells"] if c["stage"] == "tower")
        worst = max(cell["conditions"], key=lambda c: c["value"])
        scores[model].append(worst["accuracy"])

    ranks: dict[str, list[int]] = {m: [] for m in TOWERS}
    for column in range(len(axes_names)):
        order = sorted(TOWERS, key=lambda m: -scores[m][column])
        for position, model in enumerate(order, start=1):
            ranks[model].append(position)

    figure, axis = plt.subplots(figsize=(7, 4))
    for model in TOWERS:
        axis.plot(range(len(axes_names)), ranks[model], marker="o", color=COLOUR[model], label=DISPLAY[model])
    axis.set_xticks(range(len(axes_names)), axes_names, fontsize=8)
    axis.set_yticks(range(1, len(TOWERS) + 1))
    axis.invert_yaxis()
    axis.set_ylabel("rank (1 is best)")
    axis.set_title("Capability Profile: the ranking depends on the axis", fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def peaks(out: Path) -> None:
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 6.4), sharex=True)

    for model in TOWERS:
        cells = [
            c for c in json.loads((RESULTS / f"{model}_probe_attention_matched.json").read_text())["cells"]
            if c["stage"] == "tower"
        ]
        cells.sort(key=lambda c: c["relative_depth"])
        top.plot(
            [c["relative_depth"] for c in cells], [c["test_accuracy"] for c in cells],
            marker="o", markersize=3, color=COLOUR[model], label=DISPLAY[model],
        )

        cells = [
            c for c in json.loads((RESULTS / f"{model}_geometry_depth_matched.json").read_text())["cells"]
            if c["stage"] == "tower"
        ]
        cells.sort(key=lambda c: c["relative_depth"])
        depths = [c["relative_depth"] for c in cells]
        scores = [c["d1"] for c in cells]
        bottom.plot(depths, scores, marker="o", markersize=3, color=COLOUR[model], label=model)
        bottom.plot([depths[scores.index(max(scores))]], [max(scores)], "*", markersize=15,
                    color=COLOUR[model], markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    top.set_ylabel("recognition top-1")
    top.set_title("Recognition", fontsize=9.5, loc="left")
    bottom.set_ylabel("DIODE depth (d1)")
    bottom.set_xlabel("Relative Depth")
    bottom.set_title("Geometry, DIODE depth  \u00b7  stars mark each Tower's peak", fontsize=9.5, loc="left")
    for axis in (top, bottom):
        axis.axvline(1.0, color="0.45", linewidth=1, linestyle=":")
        axis.grid(alpha=0.25)
    top.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def semantic_forest(out: Path) -> None:
    rows = []
    for path in sorted(RESULTS.glob("bootstrap/semantic-*-vs-*.json")):
        payload = json.loads(path.read_text())
        protocol, difference = payload["protocol"], payload["difference"]
        label = (
            f"{payload['first']['name']} - {payload['second']['name']}\n"
            f"{protocol['readout']}, {protocol['arm']}"
        )
        rows.append((label, difference["point_estimate"], difference["lower"], difference["upper"]))

    rows.sort(key=lambda r: r[1])
    figure, axis = plt.subplots(figsize=(6.5, 0.62 * len(rows) + 1.4))
    for y, (label, point, low, high) in enumerate(rows):
        resolved = low > 0 or high < 0
        colour = COLOUR["dinov2"] if resolved else "0.55"
        axis.plot([low, high], [y, y], color=colour, linewidth=2)
        axis.plot([point], [y], "o", color=colour, markersize=6,
                  markerfacecolor=colour if resolved else "white")
    axis.axvline(0, color="0.3", linewidth=1, linestyle="--")
    axis.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_xlabel("top-1 difference, first minus second")
    axis.set_title(
        "Cross-model semantic comparisons\n"
        "hollow markers cross zero, so the ranking does not resolve",
        fontsize=10,
    )
    axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def stage_levels(out: Path) -> None:
    stages = ["tower", "merged", "projected"]
    projectors = [m for m in TOWERS if m not in ("dinov2", "siglip2")]

    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    for model in projectors:
        cells = json.loads((RESULTS / f"{model}_geometry_depth_matched.json").read_text())["cells"]
        deepest = {}
        for cell in cells:
            if cell["relative_depth"] == 1.0:
                deepest[cell["stage"]] = cell["d1"]
        values = [deepest[s] for s in stages if s in deepest]
        axis.plot(range(len(values)), values, marker="o", color=COLOUR[model], label=DISPLAY[model])

    axis.set_xticks(range(len(stages)), stages)
    axis.set_xlabel("Stage, at the deepest Relative Depth")
    axis.set_ylabel("DIODE depth (d1), matched")
    axis.set_title("The Connector raises depth decodability", fontsize=10)
    axis.grid(alpha=0.25, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def protocol(out: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    axis.set_xlim(0, 100)
    axis.set_ylim(2, 56)
    axis.axis("off")
    ink, muted, accent = "#1b1f26", "#6b7280", "#9c5a17"

    def box(x, y, width, height, label, sub=None, face="white", edge=ink, dashed=False):
        axis.add_patch(plt.Rectangle(
            (x, y), width, height, facecolor=face, edgecolor=edge, linewidth=1.1,
            linestyle="--" if dashed else "-",
        ))
        axis.text(x + width / 2, y + height / 2 + (1.4 if sub else 0), label,
                  ha="center", va="center", fontsize=8.5, color=ink)
        if sub:
            axis.text(x + width / 2, y + height / 2 - 2.6, sub,
                      ha="center", va="center", fontsize=7, color=muted)

    def arrow(x1, y1, x2, y2, colour=ink):
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.0))

    axis.text(12, 52.5, "Tower (frozen)", ha="center", fontsize=9.5, color=ink, weight="bold")
    for index in range(8):
        y = 6 + index * 5.4
        shade = 0.93 - 0.045 * (7 - index)
        axis.add_patch(plt.Rectangle((4, y), 16, 4.2, facecolor=str(shade), edgecolor=ink, linewidth=0.8))
        axis.text(12, y + 2.1, f"Relative Depth {(8 - index) / 8:.3f}", ha="center", va="center",
                  fontsize=6.8, color=ink)
        arrow(20.4, y + 2.1, 33.5, y + 2.1, muted)

    box(34, 5, 15, 44, "tower", "8 Stages cached", face="#f2f3f5")

    axis.text(61.5, 52.5, "Connector (frozen)", ha="center", fontsize=9.5, color=ink, weight="bold")
    axis.text(61.5, 49.6, "dashed where a Tower has no Projector",
              ha="center", fontsize=6.6, color=muted)
    box(54, 30, 15, 8, "merged", "2x2 regrouping", face="#f2f3f5", dashed=True)
    box(54, 15, 15, 8, "projected", "learned Projector", face="#f2f3f5", dashed=True)
    arrow(49.2, 27, 54, 34)
    arrow(49.2, 27, 54, 19)
    axis.text(61.5, 26.5, "from the deepest layer", ha="center", fontsize=6.6,
              color=muted, style="italic")

    # A bus so all three Stages visibly feed the same readouts.
    axis.plot([72, 72], [12, 44], color=accent, linewidth=1.0)
    arrow(49.2, 44, 72, 44, accent)
    arrow(69.2, 34, 72, 34, accent)
    arrow(69.2, 19, 72, 19, accent)

    axis.text(85, 52.5, "Readouts (trained)", ha="center", fontsize=9.5, color=ink, weight="bold")
    for label, sub, y in (
        ("attention / mean pool", "semantic, 1.6M params", 36),
        ("Probe3D decoder", "depth and normals", 24),
        ("cosine matching", "correspondence, training-free", 12),
    ):
        box(75, y, 22, 9, label, sub, edge=accent)
        arrow(72, y + 4.5, 75, y + 4.5, accent)

    axis.text(50, 3, "Every cell caches frozen features once, then trains only the readout.",
              ha="center", fontsize=8, color=muted, style="italic")
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    protocol(args.out / "protocol.svg")
    forest(args.out / "projector-forest.svg")
    semantic_forest(args.out / "semantic-forest.svg")
    label_budget(args.out / "label-budget.svg")
    relative_depth(args.out / "relative-depth-semantic.svg")
    peaks(args.out / "hypothesis-1-peaks.svg")
    stage_levels(args.out / "stage-levels.svg")
    occlusion(args.out / "occlusion.svg")
    capability_profile(args.out / "capability-profile.svg")
    print(f"wrote 9 figures to {args.out}")


if __name__ == "__main__":
    main()
