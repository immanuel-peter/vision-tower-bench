"""Plot full sibling geometry matrices without borrowing paired-run intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("depth", "normal"), required=True)
    parser.add_argument("--arm", choices=("raw", "matched"), required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("results/diagnostics/kimi-k31-pro-checkpoint"))
    args = parser.parse_args()
    sources = [Path("results/geometry") / f"kimi_k31_pro_geometry_{args.task}_{args.arm}.json",
               args.out / "sibling-control/geometry" / f"moonvit_v2_geometry_{args.task}_{args.arm}.json"]
    data = [json.loads(path.read_text()) for path in sources]
    metric, scale = ("d1", 100) if args.task == "depth" else ("mean_deg", 1)
    label = "Depth d1 (%)" if args.task == "depth" else "Normal error (degrees, lower is better)"
    stages = ("tower", "merged", "projected")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    parameters = None
    for index, (payload, name, color) in enumerate(zip(
            data, ("K3.1 Pro", "Fresh K3"), ("#356d93", "#858a8e"), strict=True)):
        assert payload["images"] == 771 and payload["train_images"] == 541
        assert payload["task"] == args.task and payload["capacity_matched"] == (args.arm == "matched")
        assert len(payload["cells"]) == 10
        tower = sorted((cell for cell in payload["cells"] if cell["stage"] == "tower"),
                       key=lambda cell: cell["layer_index"])
        assert [cell["layer_index"] for cell in tower] == [3, 7, 10, 14, 17, 20, 24, 27]
        axes[0].plot([cell["relative_depth"] for cell in tower],
                     [scale * cell[metric] for cell in tower], marker="o", markersize=3,
                     color=color, label=name)
        final = {cell["stage"]: cell for cell in payload["cells"] if cell["layer_index"] == 27}
        assert set(final) == set(stages)
        capacities = [final[stage]["trainable_parameters"] for stage in stages]
        if parameters is not None:
            assert capacities == parameters
        parameters = capacities
        axes[1].bar([position + (index - .5) * .32 for position in range(3)],
                    [scale * final[stage][metric] for stage in stages], width=.32,
                    color=color, label=name)
    axes[0].set(xlabel="Relative Depth", ylabel=label, title=f"{args.arm.capitalize()} capacity, all Tower depths")
    axes[1].set(xticks=range(3), xticklabels=[f"{stage.capitalize()}\n{count / 1e6:.3f}M"
                                          for stage, count in zip(stages, parameters, strict=True)],
                ylabel=label, title="Final Stages, head parameter counts below")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.legend(frameon=False, fontsize=8)
    figure.text(.02, .055, "771 DIODE images; 541/115/115 split; six seed-0 rates, three selected-rate fits, ten epochs.", fontsize=8)
    figure.text(.02, .02, "Matrix points only. Paired refits have their own point estimates and intervals. No causal training attribution.", fontsize=8)
    figure.tight_layout(rect=(0, .13, 1, 1))
    stem = args.out / f"{args.arm}-{args.task}-comparison"
    for extension in ("svg", "png", "pdf"):
        path = stem.with_suffix("." + extension)
        figure.savefig(path, dpi=180)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)
    stem.with_suffix(".figure.json").write_text(json.dumps(dict(
        inputs_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        intervals="none; actual full matrix point estimates only",
        task=args.task, arm=args.arm, geometry_feature_producer="CPU BF16 batch1 full native grids",
        fit_device="GPU0 for both depth models; GPU1 for both normal models",
        fit_threads=8, production_official_sdpa_bf16_agreement=False,
    ), indent=2) + "\n")
    print(f"Rendered two complete ten-cell matrices to {stem}")


if __name__ == "__main__":
    main()
