"""Generate paper figures and tables from repository result JSONs."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / "results"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
MODELS = ("dinov2", "siglip2", "muse_glimmer", "kimi_k26", "qwen3_5", "moonvit_v2")
NAMES = dict(zip(MODELS, ("DINOv2", "SigLIP2", "Muse Glimmer", "Kimi K2.6", "Qwen3.8", "Kimi K3")))
CONNECTORS = ("kimi_k26", "moonvit_v2", "muse_glimmer", "qwen3_5")
COLORS = dict(zip(MODELS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#666666")))
MARKERS = dict(zip(MODELS, ("o", "s", "^", "D", "v", "P")))
SOURCES = {}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8})


def read(name):
    path = RESULTS / name
    SOURCES[str(path.relative_to(PAPER.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def cells(model, kind, arm):
    return sorted((c for c in read(f"{model}_{kind}_{arm}.json")["cells"]
                   if c["stage"] == "tower"), key=lambda c: c["relative_depth"])


def save(fig, name):
    fig.savefig(FIGURES / name, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def write_table(name, spec, header, rows):
    text = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r"\\", r"\midrule"]
    text += [" & ".join(row) + r"\\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(text) + "\n")


def interval_text(d):
    return f"${d['point_estimate']:+.4f}$", f"$[{d['lower']:+.4f}, {d['upper']:+.4f}]$"


def label_and_capability():
    fig, ax = plt.subplots(figsize=(6.5, 2.7), layout="constrained")
    rows = []
    for m in MODELS:
        budget = [read(f"label-budget/{m}_probe_attention_matched_{b}.json")["cells"][0]
                  for b in ("f001", "f005", "f020", "f10")]
        ax.errorbar([.01, .05, .2, 1], [c["test_accuracy"] for c in budget],
                    yerr=[c["test_std"] for c in budget], capsize=2,
                    marker=MARKERS[m], markersize=4, color=COLORS[m], label=NAMES[m], linewidth=1.2)
        dep = max(c["d1"] for c in cells(m, "geometry_depth", "matched"))
        norm = min(c["mean_deg"] for c in cells(m, "geometry_normal", "matched"))
        nav = next(c for c in read(f"correspondence/{m}_correspondence_navi.json")["cells"] if c["stage"] == "tower")
        pert = next(c for c in read(f"perturbation/{m}_perturbation.json")["cells"] if c["stage"] == "tower")
        conditions = {c["value"]: c["accuracy"] for c in pert["conditions"]}
        rows.append([NAMES[m], f"{budget[0]['test_accuracy']:.4f}", f"{budget[-1]['test_accuracy']:.4f}",
                     f"{dep:.4f}", f"{norm:.2f}", f"{nav['recall_2cm']:.4f}", f"{conditions[.5]/conditions[0]:.4f}"])
    ax.set_xscale("log")
    ax.set_xticks([.01, .05, .2, 1], ["1%", "5%", "20%", "100%"])
    ax.set(xlabel="Labeled fraction of the training split", ylabel="Recognition top-1", ylim=(.12, .96))
    ax.grid(alpha=.2)
    ax.legend(ncol=2, loc="lower right", frameon=False)
    save(fig, "paper-label-budget.pdf")
    write_table("capability", "lrrrrrr", r"Tower & 1\% labels & Full labels & Depth $\uparrow$ & Normals $\downarrow$ & NAVI $\uparrow$ & Retention", rows)


def layers():
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.25), layout="constrained")
    kinds = (("probe_attention", "test_accuracy", "Recognition", True),
             ("geometry_depth", "d1", r"Depth $\delta_1$", True),
             ("geometry_normal", "mean_deg", "Normals (degrees)", False))
    geometry_rows = []
    before = recognition_last = 0
    for row, arm in enumerate(("matched", "raw")):
        for col, (kind, metric, title, higher) in enumerate(kinds):
            ax = axes[row, col]
            for m in MODELS:
                cc = cells(m, kind, arm)
                vals = [c[metric] for c in cc]
                best = (max if higher else min)(cc, key=lambda c: c[metric])
                ax.plot([c["relative_depth"] for c in cc], vals, color=COLORS[m], linewidth=1.2, label=NAMES[m])
                ax.plot(best["relative_depth"], best[metric], "*", color=COLORS[m], markersize=7)
                if kind.startswith("geometry"):
                    before += best["relative_depth"] < 1
                    gap = (best[metric] - cc[-1][metric]) * (1 if higher else -1)
                    geometry_rows.append([NAMES[m], "Depth" if higher else "Normals", arm,
                                          f"{best['relative_depth']:.3f}", f"{best[metric]:.4f}", f"{gap:.4f}"])
            ax.set_title(f"{title}, {arm}", fontsize=9)
            ax.set_xticks([.25, .5, .75, 1])
            ax.grid(alpha=.18)
            if row == 1:
                ax.set_xlabel("Relative depth")
    for arm in ("matched", "raw"):
        for readout in ("attention", "mean"):
            for m in MODELS:
                cc = cells(m, f"probe_{readout}", arm)
                recognition_last += max(cc, key=lambda c: c['test_accuracy'])["relative_depth"] == 1
    assert (before, recognition_last) == (23, 19), (before, recognition_last)
    for col in range(3):
        low = min(ax.get_ylim()[0] for ax in axes[:, col])
        high = max(ax.get_ylim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_ylim(low, high)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, fontsize=7.5)
    save(fig, "paper-layers.pdf")
    write_table("peaks", "lllrrr", "Tower & Task & Arm & Peak depth & Peak score & Final-layer cost", geometry_rows)


PANELS = (
    ("geometry-depth-matched", r"DIODE depth, matched ($\delta_1$)"),
    ("geometry-normal-matched", "DIODE normals, matched (degrees)"),
    ("geometry-depth-kitti-matched", r"KITTI depth, matched ($\delta_1$)"),
    ("correspondence-navi", "NAVI (recall at 2 cm)"),
    ("correspondence-scannet", "ScanNet (recall at 10 px)"),
    ("correspondence-spair", "SPair (PCK at 0.1)"),
)


def projectors():
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 4.75), layout="constrained")
    for ax, (prefix, title) in zip(axes.flat, PANELS):
        for y, m in enumerate(CONNECTORS):
            d = read(f"bootstrap/{prefix}-{m}-projected-vs-tower.json")["first_advantage"]
            resolved = d["lower"] > 0 or d["upper"] < 0
            ax.plot([d["lower"], d["upper"]], [y,y], color=COLORS[m], linewidth=1.5)
            ax.plot(d["point_estimate"], y, "o", color=COLORS[m], markersize=4,
                    markerfacecolor=COLORS[m] if resolved else "white")
        ax.axvline(0, color=".5", linewidth=.8, linestyle="--")
        ax.set_yticks(range(4), [NAMES[m] for m in CONNECTORS])
        ax.set_ylim(3.6, -.6)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.locator_params(axis="x", nbins=4)
        ax.grid(axis="x", alpha=.15)
    save(fig, "paper-projectors.pdf")
    for arm in ("matched", "raw"):
        rows, scenes = [], []
        for task in ("depth", "normal"):
            for m in CONNECTORS:
                p=read(f"bootstrap/geometry-{task}-{arm}-{m}-projected-vs-tower.json")
                rows.append([NAMES[m], "Depth" if task == "depth" else "Normals", *interval_text(p["first_advantage"])])
                if arm == "matched":
                    for scene,d in p["by_scene"].items():
                        scenes.append([NAMES[m], "Depth" if task == "depth" else "Normals", "Indoor" if scene == "indoors" else "Outdoor", *interval_text(d)])
        write_table(f"diode-{arm}", "llrr", r"Connector & Task & Difference & 95\% interval", rows)
        if scenes:
            write_table("diode-scenes", "lllrr", r"Connector & Task & Scene & Difference & 95\% interval", scenes)
    for name,prefix in (("kitti","geometry-depth-kitti-matched"), ("navi","correspondence-navi"),
                        ("scannet","correspondence-scannet"), ("spair","correspondence-spair")):
        rows = [[NAMES[m], *interval_text(read(f"bootstrap/{prefix}-{m}-projected-vs-tower.json")["first_advantage"])] for m in CONNECTORS]
        write_table(name, "lrr", r"Connector & Difference & 95\% interval", rows)


def semantic():
    rows=[]
    for path in sorted((RESULTS / "bootstrap").glob("semantic-*-vs-*.json")):
        p=read(str(path.relative_to(RESULTS)))
        def name(c):
            mid=c['model_id']
            model=next(m for m in MODELS if mid==read(f"{m}_probe_attention_matched.json")["cells"][0]['model_id'])
            return f"{NAMES[model]} {c['relative_depth']:.3f}"
        rows.append([name(p['first'])+" / "+name(p['second']), p['protocol']['readout'], p['protocol']['arm'], *interval_text(p['difference'])])
    write_table("semantic", "lllrr", r"First / second & Readout & Arm & Difference & 95\% interval", rows)
    rows=[]
    for prefix in ("geometry-depth-matched-dinov2-vs-qwen3_5", "geometry-normal-matched-dinov2-vs-siglip2"):
        p=read(f"bootstrap/{prefix}.json")
        rows.append(["DINOv2 / "+("Qwen3.8" if 'depth' in prefix else "SigLIP2"), "Depth" if 'depth' in prefix else "Normals", *interval_text(p['first_advantage'])])
    write_table("cross-geometry", "llrr", r"First / second & Task & Advantage & 95\% interval", rows)


def secondary():
    fig, axes=plt.subplots(1,2,figsize=(6.5,2.8),layout="constrained")
    corr_rows=[]
    for m in MODELS:
        p=read(f"perturbation/{m}_perturbation.json")
        cond=sorted(next(c for c in p['cells'] if c['stage']=='tower')['conditions'],key=lambda c:c['value'])
        axes[0].plot([c['value'] for c in cond],[c['accuracy'] for c in cond],color=COLORS[m],marker=MARKERS[m],markersize=3,label=NAMES[m])
        nav=read(f"correspondence/{m}_correspondence_navi.json")
        if m in CONNECTORS:
            cc={c['stage']:c for c in nav['cells']}
            axes[1].plot(range(3),[cc[s]['recall_2cm'] for s in ('tower','merged','projected')],color=COLORS[m],marker=MARKERS[m],markersize=3)
        corr=[]
        for ds,metric in (("navi","recall_2cm"),("scannet","recall_10px"),("spair","macro_pck_0.1")):
            cc=read(f"correspondence/{m}_correspondence_{ds}.json")['cells']
            corr.append(next(c for c in cc if c['stage']=='tower')[metric])
        corr_rows.append([NAMES[m],*[f"{v:.4f}" for v in corr]])
    axes[0].set(xlabel="Occluded image fraction",ylabel="Top-1 accuracy")
    axes[1].set_xticks(range(3),['Tower','Merged','Projected'])
    axes[1].set(ylabel="NAVI recall at 2 cm")
    for ax in axes: ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=6,frameon=False,fontsize=7.5)
    save(fig,"paper-secondary.pdf")
    write_table("correspondence-levels","lrrr","Tower & NAVI & ScanNet & SPair (macro)",corr_rows)


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    label_and_capability()
    layers()
    projectors()
    semantic()
    secondary()
    (PAPER / "asset-sources.json").write_text(json.dumps({"sha256":dict(sorted(SOURCES.items()))},indent=2)+"\n")
    print(f"Generated 4 figures and {len(list(TABLES.glob('*.tex')))} tables from {len(SOURCES)} result files; layer counts verified.")


if __name__ == "__main__":
    main()
