from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path("/tmp/vtb-delivery/issues/24")
out=Path("results/diagnostics/kimi-k31-pro-checkpoint")
paths=[Path("results/geometry/kimi_k31_pro_geometry_depth_raw.json"),out/"sibling-control/geometry/moonvit_v2_geometry_depth_raw.json"]
data=[json.loads(path.read_text()) for path in paths]
assert all(len(payload["cells"])==10 and payload["images"]==771 and payload["device"]=="cuda:0" for payload in data)
figure,axes=plt.subplots(1,2,figsize=(10,4.6))
stages=("tower","merged","projected")
for index,(payload,name,color) in enumerate(zip(data,("K3.1 Pro","Fresh K3"),("#356d93","#858a8e"),strict=True)):
 tower=sorted((cell for cell in payload["cells"] if cell["stage"]=="tower"),key=lambda cell:cell["layer_index"])
 axes[0].plot([cell["relative_depth"] for cell in tower],[100*cell["d1"] for cell in tower],marker="o",markersize=3,color=color,label=name)
 final={cell["stage"]:cell for cell in payload["cells"] if cell["layer_index"]==27}
 axes[1].bar([i+(index-.5)*.32 for i in range(3)],[100*final[stage]["d1"] for stage in stages],width=.32,color=color,label=name)
axes[0].set(xlabel="Relative depth",ylabel="Depth δ1 (%)",title="Raw capacity, all Tower depths")
axes[1].set(xticks=range(3),xticklabels=("Tower, 1.71M","Merged, 3.28M","Projected, 4.85M"),ylabel="Depth δ1 (%)",title="Final Stages, native head capacities")
for axis in axes:
 axis.spines[["top","right"]].set_visible(False)
 axis.grid(axis="y",alpha=.15)
 axis.legend(frameon=False,fontsize=8)
figure.text(.02,.055,"771 DIODE images; 541/115/115 split; 6 LRs at seed0, three selected-rate fits, 10 epochs. Same-device sibling controls.",fontsize=8)
figure.text(.02,.02,"Point estimates only; paired refits remain pending. Head sizes differ between raw Stages. No causal training attribution.",fontsize=8)
figure.tight_layout(rect=(0,.12,1,1))
for suffix in ("svg","png","pdf"):figure.savefig(out/f"raw-depth-comparison.{suffix}",dpi=180)
plt.close(figure)
(out/"raw-depth-comparison.figure.json").write_text(json.dumps(dict(
 inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 intervals="none; matrix point estimates only",feature_producer="CPU4 BF16 batch1 full native grids",fits="CUDA0, 8 threads for both checkpoints"),indent=2)+"\n")
print("Raw depth matrices",[[(cell["stage"],cell["d1"]) for cell in payload["cells"] if cell["layer_index"]==27] for payload in data])
