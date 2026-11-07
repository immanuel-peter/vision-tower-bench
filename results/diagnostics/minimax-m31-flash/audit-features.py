"""Validate actual cached contents and deterministic split IDs without CUDA."""
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
from vtb import cache,probe_run
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--model",required=True);a=p.parse_args()
root=Path("/tmp/vtb-delivery/issues/21");data=Path("/tmp/vtb-delivery/data")
tag="pool4" if a.dataset=="recognition" else "full"
run=root/"features"/a.dataset/f"{a.model}_448_{tag}"
images=data/"imagenet100/validation" if a.dataset=="recognition" else data/a.dataset/"val"
expected_ids=sorted(p.stem for p in images.iterdir() if p.suffix.lower() in (".png",".jpg"))
counts={"recognition":13000,"diode":771,"kitti":1000}
assert len(expected_ids)==counts[a.dataset]
expected_slices={("tower",x) for x in range(4,33,4)}|{("merged",32),("projected",32)}
if a.dataset=="kitti":expected_slices={("tower",32),("projected",32)}
assert set(cache.slices(run))==expected_slices
parity=json.loads((root/"isolated-parent-forward.json").read_text())
entries=[json.loads(line) for line in (root/"execution.jsonl").read_text().splitlines()]
runs=[row for row in entries if row.get("name")==f"extract-{a.dataset}-{a.model}" and row.get("exit_code")==0 and "source_hashes" in row]
assert runs and runs[-1]["source_hashes"]["vtb/adapters/minimax_m3.py"]==parity["adapter_source_sha256"],"No successful extraction from the independently validated isolated adapter"
report={"model":a.model,"dataset":a.dataset,"slices":{},"source_bundle_manifest":json.loads((Path("hf/MiniMax-M3.1-Flash-Vision") if a.model=="minimax_m31_flash" else root/"m3-bundle").joinpath("export_manifest.json").read_text())}
for stage,layer in cache.slices(run):
 ids=[];files=[]
 for path in cache.shards(run,stage,layer):
  with safe_open(path,framework="pt") as f:
   meta=f.metadata();tokens=f.get_tensor("tokens")
   current=meta["image_ids"].split("\n")
   assert tokens.shape[0]==len(current)
   expected_tokens=16 if tag=="pool4" else 1024 if stage=="tower" else 256
   expected_width={"tower":1280,"merged":5120,"projected":6144}[stage]
   assert tokens.shape[1:]==(expected_tokens,expected_width),(path,tokens.shape)
   assert torch.isfinite(tokens).all(),path
   assert meta["model_id"]=="immanuelpeter/"+("MiniMax-M3.1-Flash-Vision" if a.model=="minimax_m31_flash" else "MiniMax-M3-Vision")
   assert meta["resolution"]=="448" and meta["stage"]==stage and int(meta["layer_index"])==layer
   assert meta["pooled_to"]==("4" if tag=="pool4" else "none")
   files.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"images":len(current),"shape":list(tokens.shape),"dtype":str(tokens.dtype)})
   ids.extend(current)
 assert ids==expected_ids
 split=probe_run.split_indices(len(ids))
 report["slices"][f"{stage}:{layer}"]={"images":len(ids),"files":files,"ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),"split_ids":{key:[ids[i] for i in getattr(split,key).tolist()] for key in ("train","val","test")}}
report["image_inference"]="native_single_image_forward";report["execution"]=runs[-1]
out=root/f"cache-audit-{a.dataset}-{a.model}.json";out.write_text(json.dumps(report,indent=2)+"\n");print(str(out),flush=True)

