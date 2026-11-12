"""Audit every actual perturbation cache tensor and selected image without CUDA."""
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
from vtb import cache
from vtb.images import ImageFolder
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument("--model",required=True,choices=("minimax_m31_flash","minimax_m3"));a=p.parse_args()
r=Path("/tmp/vtb-delivery/issues/21");data=Path("/tmp/vtb-delivery/data")
manifest=data/"perturbation/transforms.manifest.json";protocol=json.loads(manifest.read_text());assert protocol["images"]==2000 and len(set(protocol["conditions"]))==13
selection=json.loads((r/"selection-manifest.json").read_text())["datasets"]["perturbation"];expected_ids=selection["selected_ids"]
entries=[json.loads(line) for line in (r/"execution.jsonl").read_text().splitlines()]
executions=[entry for entry in entries if entry.get("name")==f"perturbation-{a.model}" and entry.get("exit_code")==0];assert executions
execution=executions[-1];parity=json.loads((r/"isolated-parent-forward.json").read_text())
assert execution["source_hashes"]["vtb/adapters/minimax_m3.py"]==parity["adapter_source_sha256"]
assert execution["source_hashes"]["scripts/perturbation_extract.py"]==hashlib.sha256(Path("scripts/perturbation_extract.py").read_bytes()).hexdigest()
model_id="immanuelpeter/"+("MiniMax-M3.1-Flash-Vision" if a.model=="minimax_m31_flash" else "MiniMax-M3-Vision")
report={"model":a.model,"model_id":model_id,"common_image_ids":expected_ids,"selection_manifest_sha256":hashlib.sha256((r/"selection-manifest.json").read_bytes()).hexdigest(),"transform_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"execution":execution,"conditions":{},"shards":0}
extraction=json.loads((r/"features/perturbation"/f"{a.model}_extraction.json").read_text());assert {v["condition"] for v in extraction["conditions"]}==set(protocol["conditions"])
for condition in protocol["conditions"]:
 image_root=data/"imagenet100/validation" if condition=="identity" else data/"perturbation"/condition
 actual_input_ids=[p.stem for p in ImageFolder(image_root,None,2000).paths];assert actual_input_ids==expected_ids
 run=r/"features/perturbation"/condition/f"{a.model}_448_pool4"
 assert set(cache.slices(run))=={(stage,32) for stage in ("tower","merged","projected")}
 summary=json.loads((run/"summary.json").read_text());assert summary["condition"]==condition and summary["model_id"]==model_id and summary["images"]==2000 and summary["pooled_to"]==4
 item={"input_ids_sha256":hashlib.sha256("\n".join(actual_input_ids).encode()).hexdigest(),"slices":{}}
 for stage,width in (("tower",1280),("merged",5120),("projected",6144)):
  ids=[];files=[]
  for path in cache.shards(run,stage,32):
   with safe_open(path,framework="pt") as f:
    meta=f.metadata();tokens=f.get_tensor("tokens");current=meta["image_ids"].split("\n")
    assert tokens.shape==(len(current),16,width) and tokens.dtype==torch.bfloat16
    assert torch.isfinite(tokens).all(),str(path)
    assert meta["model_id"]==model_id and meta["stage"]==stage and meta["resolution"]=="448" and meta["layer_index"]=="32" and meta["pooled_to"]=="4"
    files.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"shape":list(tokens.shape),"dtype":str(tokens.dtype),"finite":True})
    ids.extend(current)
  assert ids==expected_ids and len(set(ids))==2000 and len(files)==4
  item["slices"][stage]={"files":files,"images":2000,"ids_reference":"common_image_ids"};report["shards"]+=len(files)
 assert len(list(run.glob("*.safetensors")))==12
 report["conditions"][condition]=item
assert report["shards"]==156
report["complete"]=True
out=r/f"perturbation-cache-audit-{a.model}.json";out.write_text(json.dumps(report,indent=2)+"\n")
print(a.model,"all156shards/13conditions/3Stages/exact2000IDs/BF16/finite/hash/sourceepoch validated")
