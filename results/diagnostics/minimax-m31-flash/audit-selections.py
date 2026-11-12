"""Reconstruct exact deterministic selections from audited cache order and hashed runners."""
import hashlib,json
from pathlib import Path
import torch
from vtb.probe_run import split_indices,subsample_train
torch.set_num_threads(2)
r=Path("/tmp/vtb-delivery/issues/21")
labels_path=Path("/tmp/vtb-delivery/data/imagenet100/validation_labels.json")
label_map=json.loads(labels_path.read_text())["labels"]
report={"scope":"Deterministically reconstructed using the exact runner selection functions and audited cache order; primary payloads retain counts, not runtime subset ID traces.","runner_sha256":hashlib.sha256(Path("vtb/probe_run.py").read_bytes()).hexdigest(),"labels_sha256":hashlib.sha256(labels_path.read_bytes()).hexdigest(),"datasets":{}}
executions=[json.loads(line) for line in (r/"execution.jsonl").read_text().splitlines()]
latest={entry["name"]:entry for entry in executions if entry.get("exit_code")==0 and entry.get("name","").startswith(("recognition-","label-budget-","standardized-"))}
assert len(latest)==12,len(latest)
assert all(entry["source_hashes"]["vtb/probe_run.py"]==report["runner_sha256"] for entry in latest.values())
report["torch_version"]=torch.__version__
report["successful_semantic_execution_sources"]={name:{"start_utc":entry["start_utc"],"runner_sha256":entry["source_hashes"]["vtb/probe_run.py"]} for name,entry in latest.items()}
for dataset,count in (("recognition",13000),("diode",771),("kitti",1000)):
 common=None;sources={}
 for model in ("minimax_m31_flash","minimax_m3"):
  path=r/f"cache-audit-{dataset}-{model}.json";d=json.loads(path.read_text());sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  for c in d["slices"].values():
   if common is None:common=c["split_ids"]
   assert c["split_ids"]==common
 ids=sorted(common["train"]+common["val"]+common["test"]);assert len(ids)==len(set(ids))==count
 split=split_indices(count)
 assert {k:[ids[i] for i in getattr(split,k).tolist()] for k in common}==common
 item={"cache_audit_sha256":sources,"split_seed":0,"splits":common}
 if dataset=="recognition":
  labels=torch.tensor([label_map[i] for i in ids]);item["label_budgets"]={}
  for fraction,expected in ((.01,100),(.05,455),(.2,1820),(1.,9100)):
   chosen=subsample_train(split,labels,fraction);selected=[ids[i] for i in chosen.train.tolist()]
   assert len(selected)==len(set(selected))==expected and set(selected)<=set(common["train"])
   assert torch.equal(chosen.val,split.val) and torch.equal(chosen.test,split.test)
   assert len({label_map[i] for i in selected})==100
   item["label_budgets"][str(fraction)]={"train_ids":selected,"ids_sha256":hashlib.sha256("\n".join(selected).encode()).hexdigest(),"classes":100,"shared_primary_and_standardized":True}
 report["datasets"][dataset]=item
ids=sorted(label_map)[:2000];split=split_indices(len(ids))
report["datasets"]["perturbation"]={"selected_ids":ids,"splits":{k:[ids[i] for i in getattr(split,k).tolist()] for k in ("train","val","test")},"split_seed":0,"selection":"First2000sortedImageNet100IDs, all13conditions, bothcheckpoints"}
(r/"selection-manifest.json").write_text(json.dumps(report,indent=2)+"\n")
print("Recorded exact recognition label-budget subsets and four dataset splits; model/stage orders agree")
