"""Independently check paired correspondence/perturbation arrays against source payloads."""
import argparse,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
import torch
from vtb.correspondence import paired_bootstrap
from vtb.semantic_bootstrap import paired_image_bootstrap
from scripts.cross_run_bootstrap import class_macro_bootstrap
torch.set_num_threads(2)
root=Path("/tmp/vtb-delivery/issues/21");results=root/"measurements";flash="minimax_m31_flash";models=(flash,"minimax_m3")
args=SimpleNamespace(resamples=10000,confidence=.95,bootstrap_seed=0)
zero={key for key,value in json.loads((root/"spair-visibility-audit.json").read_text())["source_visibility"].items() if value["visible_keypoints"]==0}
parser=argparse.ArgumentParser();parser.add_argument("--section",choices=("all","correspondence","remaining"),default="all");phase=parser.parse_args()
report={}
def read(path):return json.loads(path.read_text())
def interval(actual,expected):
 for key in ("point_estimate","lower","upper"):
  assert math.isfinite(actual[key]) and abs(actual[key]-expected[key])<1e-7,(key,actual[key],expected[key])
def by_stage(payload,stage):return next(c for c in payload["cells"] if c["stage"]==stage)
if phase.section in ("all","correspondence"):
 for dataset in ("navi","scannet","spair"):
  metric={"navi":"recall_2cm","scannet":"recall_10px","spair":"pck_0.1"}[dataset]
  source={model:read(results/"correspondence"/f"{model}_correspondence_{dataset}.json") for model in models}
  for model in models:
   path=results/"bootstrap"/f"correspondence-{dataset}-{model}-projected-vs-tower.json";d=read(path)
   rows=source[model]["measurements"]
   valid=[r for r in rows if all(math.isfinite(r["cells"][stage+":32"][metric]) for stage in ("projected","tower"))]
   excluded={r["pair_id"] for r in rows}-{r["pair_id"] for r in valid}
   assert d["paired_selection"]["finite_intersection_ids"]==[r["pair_id"] for r in valid]
   assert {r["pair_id"] for r in d["paired_selection"]["excluded_union"]}==excluded
   if dataset=="spair":assert excluded==zero
   x=torch.tensor([r["cells"]["projected:32"][metric] for r in valid]);y=torch.tensor([r["cells"]["tower:32"][metric] for r in valid])
   interval(d["first_advantage"],paired_bootstrap(x,y))
   if dataset=="spair":
    pairs=[{"class":r["class"],"projected":r["cells"]["projected:32"][metric],"tower":r["cells"]["tower:32"][metric]} for r in valid]
    interval(d["class_macro_first_advantage"],class_macro_bootstrap(pairs,("projected","tower"),args))
   report[path.name]={"selected":len(rows),"finite_intersection":len(valid),"excluded_union":sorted(excluded),"interval_recomputed":True}
  for stage in ("tower","merged","projected"):
   path=results/"bootstrap"/f"correspondence-{dataset}-m31-vs-m3-{stage}.json";d=read(path)
   a={r["pair_id"]:r for r in source[flash]["measurements"]};b={r["pair_id"]:r for r in source["minimax_m3"]["measurements"]}
   assert a.keys()==b.keys()
   valid={key for key in a if all(math.isfinite(r[key]["cells"][stage+":32"][metric]) for r in (a,b))}
   excluded=set(a)-valid
   assert {r["pair_id"] for r in d["measurements"]}==valid and len(d["measurements"])==len(valid)
   assert set(d["excluded_nonfinite_pair_ids"])==excluded
   if dataset=="spair":assert excluded==zero
   for row in d["measurements"]:
    assert row[flash]==a[row["pair_id"]]["cells"][stage+":32"][metric]
    assert row["minimax_m3"]==b[row["pair_id"]]["cells"][stage+":32"][metric]
   x=torch.tensor([r[flash] for r in d["measurements"]]);y=torch.tensor([r["minimax_m3"] for r in d["measurements"]])
   interval(d["difference"],paired_bootstrap(x,y))
   if dataset=="spair":interval(d["class_macro_difference"],class_macro_bootstrap(d["measurements"],(flash,"minimax_m3"),args))
   report[path.name]={"finite_intersection":len(valid),"excluded_union":sorted(excluded),"arrays_match_sources":True}
if phase.section in ("all","remaining"):
 source={model:read(results/"perturbation"/f"{model}_perturbation.json") for model in models}
 for model in models:
  path=results/"bootstrap"/f"perturbation-{model}.json";d=read(path)
  first=by_stage(source[model],"projected");second=by_stage(source[model],"tower")
  for condition,value in d["conditions"].items():
   a={r["image_id"]:r["perturbed"] for r in first["measurements"][condition]};b={r["image_id"]:r["perturbed"] for r in second["measurements"][condition]}
   assert a.keys()==b.keys() and set(value["image_ids"])==set(a) and len(value["image_ids"])==len(a)
   x=torch.tensor([a[i] for i in value["image_ids"]]).T;y=torch.tensor([b[i] for i in value["image_ids"]]).T
   assert x.tolist()==value["projected_by_seed"] and y.tolist()==value["tower_by_seed"]
   interval(value["difference"],paired_image_bootstrap(x,y))
  assert len(d["conditions"])==13
  report[path.name]={"conditions":13,"test_images":len(a),"arrays_match_sources":True}
 for stage in ("tower","merged","projected"):
  path=results/"bootstrap"/f"perturbation-m31-vs-m3-{stage}.json";d=read(path)
  for condition,value in d["conditions"].items():
   a={r["image_id"]:r for r in by_stage(source[flash],stage)["measurements"][condition]}
   b={r["image_id"]:r for r in by_stage(source["minimax_m3"],stage)["measurements"][condition]}
   assert a.keys()==b.keys() and {r["image_id"] for r in value["measurements"]}==set(a) and len(value["measurements"])==len(a)
   for row in value["measurements"]:
    for model,original in ((flash,a),("minimax_m3",b)):
     assert row[model]==original[row["image_id"]]["perturbed"]
     assert row[model+"_clean"]==original[row["image_id"]]["clean"]
   x=torch.tensor([r[flash] for r in value["measurements"]]).T;y=torch.tensor([r["minimax_m3"] for r in value["measurements"]]).T
   interval(value["difference"],paired_image_bootstrap(x,y))
  assert len(d["conditions"])==13
  clean_x=torch.tensor([r[flash+"_clean"] for r in value["measurements"]]).T
  clean_y=torch.tensor([r["minimax_m3_clean"] for r in value["measurements"]]).T
  interval(d["clean"]["difference"],paired_image_bootstrap(clean_x,clean_y))
  report[path.name]={"conditions":13,"arrays_match_sources":True}
if phase.section=="correspondence":
 target=root/"paired-source-validation-correspondence.json"
else:
 if phase.section=="remaining":report=json.loads((root/"paired-source-validation-correspondence.json").read_text())|report
 target=root/"paired-source-validation.json"
target.write_text(json.dumps(report,indent=2)+"\n")
print("Independently validated",len(report),"paired source payloads",flush=True)
