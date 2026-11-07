"""Content-level validation of primary and paired measurement inventories."""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from vtb import probe_run
from vtb.semantic_bootstrap import paired_image_bootstrap
from vtb.geometry_bootstrap import paired_metric_bootstrap
from vtb.correspondence import paired_bootstrap
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument("--partial",action="store_true");p.add_argument("--model",choices=("minimax_m31_flash","minimax_m3"),default="minimax_m31_flash");a=p.parse_args()
root=Path("/tmp/vtb-delivery/issues/21");results=root/"measurements";flash=a.model
model_id="immanuelpeter/"+("MiniMax-M3.1-Flash-Vision" if flash=="minimax_m31_flash" else "MiniMax-M3-Vision")
slices={("tower",x) for x in range(4,33,4)}|{("merged",32),("projected",32)}
rates_sem={1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1,.3,1.}
rates_geo={1e-4,3e-4,1e-3,3e-3,1e-2,3e-2}
report={"files":{},"missing":[],"counts":{}}
split_manifest=json.loads((root/"selection-manifest.json").read_text())["datasets"]
label_map=json.loads(Path("/tmp/vtb-delivery/data/imagenet100/validation_labels.json").read_text())["labels"]
known_labels=set(label_map.values())
def read(path,kind):
 if not path.exists():
  report["missing"].append(str(path));return None
 d=json.loads(path.read_text())
 report["files"][str(path.relative_to(results))]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"kind":kind}
 return d
def finite(x):
 assert isinstance(x,(int,float)) and math.isfinite(x),x
def search(cell,rates):
 assert {float(k) for k in cell["learning_rate_search"]}==rates
 assert float(cell["learning_rate"]) in rates
 for score in cell["learning_rate_search"].values():finite(score)
def count(kind,n):report["counts"][kind]=report["counts"].get(kind,0)+n
for readout in ("attention","mean"):
 for arm in ("raw","matched"):
  d=read(results/"recognition"/f"{flash}_probe_{readout}_{arm}.json","recognition")
  if d:
   assert d["images"]==13000 and d["train_images"]==9100 and d["label_fraction"]==1
   assert {(c["stage"],c["layer_index"]) for c in d["cells"]}==slices and len(d["cells"])==10
   for c in d["cells"]:
    search(c,rates_sem);assert c["seeds"]==3 and c["model_id"]==model_id
    assert c["train_images"]==9100 and c["full_train_images"]==9100
    for key in ("val_accuracy","test_accuracy","test_std"):finite(c[key])
    assert 0<=c["test_accuracy"]<=1
   count("recognition_cells",10)
for task in ("depth","normal"):
 for arm in ("raw","matched"):
  d=read(results/"geometry"/f"{flash}_geometry_{task}_{arm}.json","geometry")
  if d:
   assert d["images"]==771 and d["train_images"]==541 and d["epochs"]==10 and d["seeds"]==3
   assert set(d["learning_rates"])==rates_geo and d["capacity_matched"]==(arm=="matched")
   assert {(c["stage"],c["layer_index"]) for c in d["cells"]}==slices and len(d["cells"])==10
   for c in d["cells"]:
    search(c,rates_geo);assert c["model_id"]==model_id and c["seeds"]==3
    assert len(c["per_seed"])==3
    for seed in c["per_seed"]:
     assert seed["images"]==115
     finite(seed["d1" if task=="depth" else "mean_deg"])
    finite(c["d1" if task=="depth" else "mean_deg"])
   count("geometry_cells",10)
for readout in ("attention","mean"):
 for tag,fraction,train in (("f001",.01,100),("f005",.05,455),("f020",.2,1820),("f10",1.,9100)):
  d=read(results/"label-budget"/f"{flash}_probe_{readout}_matched_{tag}.json","label_budget")
  if d:
   assert d["images"]==13000 and d["label_fraction"]==fraction and d["train_images"]==train and d["full_train_images"]==9100
   assert len(d["cells"])==1
   c=d["cells"][0];assert c["stage"]=="tower" and c["layer_index"]==32 and c["seeds"]==3 and c["model_id"]==model_id
   search(c,rates_sem);finite(c["test_accuracy"]);count("label_budget_cells",1)
for tag,fraction,train in (("f001",.01,100),("f005",.05,455),("f02",.2,1820),("f10",1.,9100)):
 d=read(results/"standardized"/f"{flash}_probe_attention_matched_{tag}.json","standardized")
 if d:
  assert d["images"]==13000 and d["label_fraction"]==fraction and d["train_images"]==train
  assert len(d["cells"])==1
  c=d["cells"][0];assert c["stage"]=="tower" and c["layer_index"]==32 and c["model_id"]==model_id and c["seeds"]==3
  assert c["reducer_diagnostic"]=="channel_standardization_sample_std_correction1"
  search(c,rates_sem);finite(c["test_accuracy"]);count("standardized_cells",1)
selection=json.loads((root/"correspondence-selection.json").read_text())
for model in dict.fromkeys((flash,"minimax_m3")):
 for dataset in ("navi","scannet","spair"):
  d=read(results/"correspondence"/f"{model}_correspondence_{dataset}.json","correspondence")
  if not d:continue
  assert d["model_id"]=="immanuelpeter/"+("MiniMax-M3.1-Flash-Vision" if model=="minimax_m31_flash" else "MiniMax-M3-Vision")
  assert d["model"]==model and d["resolution"]==448 and d["evaluation_side"]==64 and d["num_correspondences"]==1000
  assert d["pairs"]==selection[dataset]["pairs"] and len(d["measurements"])==d["pairs"]
  ids=[r["pair_id"] for r in d["measurements"]]
  assert ids==selection[dataset]["ids"] and len(set(ids))==len(ids)
  if dataset=="spair":
   source=json.loads((root/"spair-visibility-audit.json").read_text())["source_visibility"]
   zero_ids={key for key,value in source.items() if value["visible_keypoints"]==0}
   stage_exclusions={}
   for stage in ("tower","merged","projected"):
    actual_zero=set()
    for row in d["measurements"]:
     value=row["cells"][stage+":32"]
     expected_count=source[row["pair_id"]]["visible_keypoints"]
     assert value["keypoints"]==expected_count,(model,row["pair_id"],stage)
     if expected_count:
      assert math.isfinite(value["pck_0.1"]),(model,row["pair_id"],"nonfinite on nonempty pair")
     else:
      assert math.isnan(value["pck_0.1"]),(model,row["pair_id"],"unexpected empty-pair metric")
      actual_zero.add(row["pair_id"])
    assert actual_zero==zero_ids
    stage_exclusions[stage]=sorted(actual_zero)
   report["files"][str((results/"correspondence"/f"{model}_correspondence_{dataset}.json").relative_to(results))]["source_verified_exclusions"]=stage_exclusions
  else:
   for row in d["measurements"]:
    for cell in row["cells"].values():
     if cell["matches"]>0:
      assert all(math.isfinite(v) for key,v in cell.items() if key!="matches"),(model,dataset,row["pair_id"])

  assert {c["stage"] for c in d["cells"]}=={"tower","merged","projected"} and len(d["cells"])==3
  metric={"navi":"recall_2cm","scannet":"recall_10px","spair":"pck_0.1"}[dataset]
  for c in d["cells"]:
   finite(c[metric])
   vals=[r["cells"][f'{c["stage"]}:32'][metric] for r in d["measurements"]]
   valid=[x for x in vals if math.isfinite(x)]
   assert valid and abs(sum(valid)/len(valid)-c[metric])<2e-6
   report["files"][str((results/"correspondence"/f"{model}_correspondence_{dataset}.json").relative_to(results))].setdefault("nonfinite_pairs",{})[c["stage"]]=len(vals)-len(valid)
  count("correspondence_cells" if model==flash else "sibling_correspondence_cells",3)
for model in dict.fromkeys((flash,"minimax_m3")):
 d=read(results/"perturbation"/f"{model}_perturbation.json","perturbation")
 if not d:continue
 assert {c["stage"] for c in d["cells"]}=={"tower","merged","projected"}
 assert all(c["model_id"]=="immanuelpeter/"+("MiniMax-M3.1-Flash-Vision" if model=="minimax_m31_flash" else "MiniMax-M3-Vision") for c in d["cells"])
 assert d["images"]==2000 and d["protocol"]["seeds"]==3 and d["protocol"]["epochs"]==20
 assert set(d["protocol"]["learning_rates"])==rates_sem and len(d["cells"])==3
 reference_ids=None
 for c in d["cells"]:
  assert c["train_images"]==1400 and c["test_images"]==300 and c["validation_images"]==300
  search(c,rates_sem);assert len(c["conditions"])==13 and len(c["measurements"])==13
  for condition in c["conditions"]:
   finite(condition["accuracy"]);assert len(condition["accuracy_by_seed"])==3
   rows=c["measurements"][condition["condition"]]
   assert len(rows)==300
   ids=[r["image_id"] for r in rows];assert len(set(ids))==300
   if reference_ids is None:reference_ids=ids
   assert ids==reference_ids
   assert all(len(r["clean"])==len(r["perturbed"])==3 for r in rows)
   assert all(value in (0,1) for row in rows for key in ("clean","perturbed") for value in row[key])
   expected_ids=sorted(label_map)[:2000]
   expected_test=[expected_ids[i] for i in probe_run.split_indices(2000).test.tolist()]
   assert ids==expected_test
   if condition["condition"]=="identity":assert all(row["clean"]==row["perturbed"] for row in rows)
   measured=torch.tensor([r["perturbed"] for r in rows]).float().mean().item()
   assert abs(measured-condition["accuracy"])<1e-6
   observed=torch.tensor([r["perturbed"] for r in rows]).float().mean(dim=0)
   assert all(abs(x-y)<1e-6 for x,y in zip(observed.tolist(),condition["accuracy_by_seed"]))
   clean=torch.tensor([r["clean"] for r in rows]).float().mean(dim=1)
   perturbed=torch.tensor([r["perturbed"] for r in rows]).float().mean(dim=1)
   interval=paired_bootstrap(clean,perturbed)
   for key in ("point_estimate","lower","upper"):
    assert abs(interval[key]-condition["clean_minus_condition"][key])<1e-7,(model,c["stage"],condition["condition"],key)
 count("perturbation_conditions" if model==flash else "sibling_perturbation_conditions",39)
d=read(results/"kitti"/f"{flash}_geometry_depth_matched.json","kitti")
if d:
 assert d["images"]==1000 and d["train_images"]==700 and d["epochs"]==10 and d["seeds"]==3
 assert {(c["stage"],c["layer_index"]) for c in d["cells"]}=={("tower",32),("projected",32)}
 for c in d["cells"]:
  search(c,rates_geo);assert len(c["per_seed"])==3 and c["model_id"]==model_id
  for seed in c["per_seed"]:assert seed["images"]==150
  finite(c["d1"])
 count("kitti_cells",2)
d=read(results/"throughput"/f"{flash}.json","throughput")
if d:
 assert d["images"]>=768 and d["resolution"]==448 and d["configured_patch_size"]==14 and d["workers"]<=2 and d["model_id"]==model_id
 finite(d["images_per_second"]);assert d["images_per_second"]>0;count("throughput_records",1)
# Every paired rerun must retain both sides under distinct names.
expected_bootstrap=[]
for task in ("depth","normal"):
 for arm in ("matched","raw"):
  expected_bootstrap.append(f"geometry-{task}-{arm}-{flash}-projected-vs-tower.json")
  expected_bootstrap.extend(f"geometry-{task}-diode-{arm}-m31-vs-m3-{stage}.json" for stage in ("tower","merged","projected"))
expected_bootstrap.append(f"geometry-depth-kitti-matched-{flash}-projected-vs-tower.json")
expected_bootstrap.extend(f"geometry-depth-kitti-matched-m31-vs-m3-{stage}.json" for stage in ("tower","projected"))
expected_bootstrap.extend(f"semantic-m31-vs-m3-{stage}.json" for stage in ("tower","merged","projected"))
expected_bootstrap.append(f"semantic-attention-matched-{flash}-projected-minus-tower.json")
expected_bootstrap.extend(f"correspondence-{ds}-{flash}-projected-vs-tower.json" for ds in ("navi","scannet","spair"))
expected_bootstrap.extend(f"correspondence-{ds}-m31-vs-m3-{stage}.json" for ds in ("navi","scannet","spair") for stage in ("tower","merged","projected"))
expected_bootstrap.extend(f"perturbation-m31-vs-m3-{stage}.json" for stage in ("tower","merged","projected"))
expected_bootstrap.append(f"perturbation-{flash}.json")
for name in expected_bootstrap:
 d=read(results/"bootstrap"/name,"bootstrap")
 if not d:continue
 if name.startswith("geometry-"):
  first,second=d["first"]["name"],d["second"]["name"]
  assert first!=second
  rows=d["measurements"];ids=[r["image_id"] for r in rows];assert len(ids)==len(set(ids))==d["protocol"]["test_images"]
  dataset="kitti" if "-kitti-" in name else "diode"
  assert set(ids)==set(split_manifest[dataset]["splits"]["test"]),(name,"test split mismatch")
  report["files"]["bootstrap/"+name]["test_split_reference"]=dataset
  report["files"]["bootstrap/"+name]["test_ids_sha256"]=hashlib.sha256("\n".join(ids).encode()).hexdigest()
  x=torch.tensor([r[first] for r in rows]).T;y=torch.tensor([r[second] for r in rows]).T
  assert x.shape==y.shape and x.shape[0]==3 and torch.isfinite(x).all() and torch.isfinite(y).all()
  expected=paired_metric_bootstrap(x,y,higher_is_better=d["protocol"]["higher_is_better"])
  actual=d["first_advantage"]
  assert abs(x.mean().item()-d["first"]["mean_metric"])<1e-5
  assert abs(y.mean().item()-d["second"]["mean_metric"])<1e-5
 elif name.startswith("semantic-"):
  first,second=d["first"]["name"],d["second"]["name"];assert first!=second
  rows=d["predictions"];ids=[r["image_id"] for r in rows];assert len(ids)==len(set(ids))==1950
  assert set(ids)==set(split_manifest["recognition"]["splits"]["test"]),(name,"test split mismatch")
  assert all(row["label"]==label_map[row["image_id"]] for row in rows)
  assert all(value in known_labels for row in rows for key in (first,second) for value in row[key])
  report["files"]["bootstrap/"+name]["test_split_reference"]="recognition"
  report["files"]["bootstrap/"+name]["test_ids_sha256"]=hashlib.sha256("\n".join(ids).encode()).hexdigest()
  labels=torch.tensor([r["label"] for r in rows])
  x=torch.tensor([r[first] for r in rows]).T==labels
  y=torch.tensor([r[second] for r in rows]).T==labels
  assert x.shape==y.shape==(3,1950)
  expected=paired_image_bootstrap(x,y);actual=d["difference"]
  assert abs(x.float().mean().item()-d["first"]["mean_accuracy"])<1e-6
  assert abs(y.float().mean().item()-d["second"]["mean_accuracy"])<1e-6
 else:
  expected=actual=None # Independently validated from paired source files during CPU postprocessing.
 if expected is not None:
  for key in ("point_estimate","lower","upper"):
   assert abs(actual[key]-expected[key])<1e-7,(name,key,actual[key],expected[key])
 count("bootstrap_payloads",1)
report["complete"]=not report["missing"]
report["expected_bootstrap_payloads"]=len(expected_bootstrap)
(root/(("measurement-validation-partial" if a.partial else "measurement-validation")+("-minimax_m3" if flash=="minimax_m3" else "")+".json")).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"counts":report["counts"],"missing":len(report["missing"]),"complete":report["complete"]}),flush=True)
if not a.partial:assert report["complete"],report["missing"]

