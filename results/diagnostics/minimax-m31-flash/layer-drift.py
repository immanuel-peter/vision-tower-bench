"""Compute concatenated-weight layer drift and render its standalone figure on CPU."""
import json,math
from pathlib import Path
import torch
from safetensors.torch import load_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(2)
root=Path("/tmp/vtb-delivery/issues/21");out=Path("results/diagnostics/minimax-m31-flash")
groups={}
for part,filename in (("tower","model.safetensors"),("projector","projector.safetensors")):
 a=load_file(root/"m3-bundle"/filename);b=load_file(Path("hf/MiniMax-M3.1-Flash-Vision")/filename)
 for name,t in a.items():
  group="layer_"+str(int(name.split(".")[1])+1) if name.startswith("layers.") else name.split(".")[0]
  key=part+"."+group
  row=groups.setdefault(key,{"dot":0.,"m3_squared_norm":0.,"flash_squared_norm":0.,"tensors":0,"parameters":0})
  x=t.reshape(-1);y=b[name].reshape(-1)
  for start in range(0,x.numel(),1_048_576):
   xx=x[start:start+1_048_576].double();yy=y[start:start+1_048_576].double()
   row["dot"]+=torch.dot(xx,yy).item()
   row["m3_squared_norm"]+=torch.dot(xx,xx).item()
   row["flash_squared_norm"]+=torch.dot(yy,yy).item()
  row["tensors"]+=1;row["parameters"]+=t.numel()
 for key,row in list(groups.items()):
  if not key.startswith(part+"."):continue
  row["cosine"]=row["dot"]/math.sqrt(row["m3_squared_norm"]*row["flash_squared_norm"])
  row["relative_l2"]=math.sqrt(max(0,row["m3_squared_norm"]+row["flash_squared_norm"]-2*row["dot"])/row["m3_squared_norm"])
 del a,b
out.mkdir(parents=True,exist_ok=True)
(out/"layer-drift.json").write_text(json.dumps({"method":"Concatenated weights per layer or module, float64 dot products and norms; M3 denominator for relative L2.","groups":groups},indent=2)+"\n")
layers=list(range(1,33))
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
axes[0].plot(layers,[groups[f"tower.layer_{i}"]["cosine"] for i in layers],color="#393b79",marker=".",linewidth=1.5)
axes[0].set(xlabel="Tower layer",ylabel="Weight cosine similarity",title="M3.1 Flash vs. M3")
axes[1].plot(layers,[100*groups[f"tower.layer_{i}"]["relative_l2"] for i in layers],color="#9e9ac8",marker=".",linewidth=1.5)
axes[1].set(xlabel="Tower layer",ylabel="Relative L2 difference (%)",title="All tensors in each layer")
for ax in axes:ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
(root/"figures").mkdir(exist_ok=True)
fig.savefig(root/"figures/minimax-m31-weight-drift.png",dpi=180)
plt.close(fig)
print("Computed",len(groups),"groups",flush=True)

