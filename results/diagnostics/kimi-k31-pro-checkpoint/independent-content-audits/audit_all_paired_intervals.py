"""Replay every required interval from actual rows after the GPU phase releases."""
import datetime,hashlib,json,os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.cross_run_bootstrap import correspondence,perturbation
from scripts.correspondence_bootstrap import interval,PRIMARY
from scripts.validate_kimi_results import finite_numbers,validate_correspondence,validate_pairing,GEOMETRY_RATES
from vtb.probe_run import LEARNING_RATES
from vtb.geometry_bootstrap import paired_metric_bootstrap
from vtb.semantic_bootstrap import paired_image_bootstrap

root=Path("/tmp/vtb-delivery/issues/24")
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
release=json.loads((root/"remaining-gpu-release.json").read_text())
assert release["passed"] and release["physical_gpu"]==0 and release["exclusive_lock_released"]
book=json.loads((root/"completed-inventory.json").read_text())
expected=json.loads((root/"expected-inventory.json").read_text())
assert book["completed_primary_cells"]==286 and book["completed_paired_cells"]==102
read=lambda path:json.loads(Path(path).read_text())
groups=defaultdict(list)
for record in book["completed_paired"]:groups[record["source"]].append(record)
results=[]
for source,records in groups.items():
 path=Path(source);payload=read(path);task=records[0]["cell"]["task"]
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 assert all(record["sha256"]==digest and record["cell"]["task"]==task for record in records)
 assert finite_numbers(payload)
 args=SimpleNamespace(resamples=10000,confidence=.95,bootstrap_seed=0,metric=None)
 if task=="recognition":
  assert len(records)==1
  key=records[0]["cell"]
  for side in ("first","second"):
   assert payload[side]["stage"]==key[side+"_stage"]
   assert payload[side]["layer_index"]==key[side+"_layer"]
  protocol=payload["protocol"]
  assert protocol["readout"]==key["readout"]=="attention" and protocol["arm"]==key["arm"]=="matched"
  assert protocol["epochs"]==20 and protocol["seeds"]==3 and protocol["test_images"]==1950
  for side in ("first","second"):
   assert {float(rate) for rate in payload[side]["learning_rate_search"]}==set(LEARNING_RATES)
  threads=protocol.get("bootstrap_threads",protocol["threads"]);torch.set_num_threads(threads)
  rows=payload["predictions"]
  assert [row["image_id"] for row in rows]==read(root/"splits-imagenet100.json")["test"]
  actual_labels=read("/tmp/vtb-delivery/data/imagenet100/validation_labels.json")["labels"]
  assert [row["label"] for row in rows]==[actual_labels[row["image_id"]] for row in rows]
  labels=torch.tensor([row["label"] for row in rows]);names=[payload[key]["name"] for key in ("first","second")]
  correct=[torch.tensor([row[name] for row in rows]).T==labels for name in names]
  assert all(tensor.shape==(3,1950) for tensor in correct)
  first_model="kimi_k31_pro"
  second_model="moonvit_v2" if records[0]["cell"]["comparison"]=="kimi_k31_pro-minus-moonvit_v2" else "kimi_k31_pro"
  fit_cells=[]
  for side,model in (("first",first_model),("second",second_model)):
   matrix=read(root/"measurements/recognition"/f"{model}_probe_attention_matched.json")
   summary=payload[side]
   fit=next(cell for cell in matrix["cells"] if cell["stage"]==summary["stage"] and cell["layer_index"]==summary["layer_index"])
   assert [list(values) for values in zip(*(row[summary["name"]] for row in rows))]==fit["test_predictions_by_seed"]
   fit_cells.append(fit)
  assert fit_cells[0]["fit_device"]==fit_cells[1]["fit_device"] and fit_cells[0]["fit_threads"]==fit_cells[1]["fit_threads"]
  assert paired_image_bootstrap(*correct)==payload["difference"]
 elif task=="perturbation":
  assert len(records)==13
  torch.set_num_threads(4)
  protocol=payload["protocol"];args.stage=protocol["first_stage"];args.second_stage=protocol["second_stage"]
  assert all(record["cell"]["first_stage"]==args.stage and record["cell"]["second_stage"]==args.second_stage for record in records)
  names=tuple(protocol["difference"].split("_minus_",1))
  replay=perturbation(read(protocol["first_source"]),read(protocol["second_source"]),names,args)
  assert all(payload[key]==value for key,value in replay.items())
  assert {record["cell"]["condition"] for record in records}=={"identity",*payload["conditions"]}
 elif task in {"depth","normal","kitti"}:
  assert len(records)==1
  torch.set_num_threads(8)
  key=records[0]["cell"];protocol=payload["protocol"]
  assert protocol["task"]==("depth" if task=="kitti" else task)
  assert protocol["arm"]==key.get("arm","matched")
  assert set(protocol["learning_rates"])==GEOMETRY_RATES
  splits=read(root/("splits-kitti.json" if task=="kitti" else "splits-diode.json"))
  assert [row["image_id"] for row in payload["measurements"]]==splits["test"]
  assert protocol["higher_is_better"]==(task!="normal") and protocol["seeds"]==3 and protocol["epochs"]==10
  tensors=[]
  for side in ("first","second"):
   summary=payload[side]
   assert summary["stage"]==key[side+"_stage"]
   assert summary["layer_index"]==key.get(side+"_layer",27)
   search=summary["learning_rate_search"]
   assert len(search)==6 and {float(rate) for rate in search}==GEOMETRY_RATES
   selected=next(value for rate,value in search.items() if float(rate)==summary["learning_rate"])
   assert selected==summary["validation_score"]==(min(search.values()) if task=="normal" else max(search.values()))
   name=summary["name"]
   tensor=torch.tensor([row[name] for row in payload["measurements"]]).T.contiguous()
   assert tensor.shape==(3,len(splits["test"])) and torch.isfinite(tensor).all()
   assert torch.equal(tensor.mean(1),torch.tensor(summary["seed_metrics"]))
   assert tensor.mean(1).mean().item()==summary["mean_metric"]
   tensors.append(tensor)
  assert paired_metric_bootstrap(*tensors,higher_is_better=protocol["higher_is_better"])==payload["first_advantage"]
  for scene,value in payload["by_scene"].items():
   positions=[i for i,row in enumerate(payload["measurements"]) if row["scene"]==scene]
   assert value==paired_metric_bootstrap(*(tensor[:,positions] for tensor in tensors),higher_is_better=protocol["higher_is_better"])|{"test_images":len(positions)}
 elif task in {"navi","scannet","spair"}:
  assert len(records)==1
  torch.set_num_threads(4 if task=="navi" else 8)
  key=records[0]["cell"];selected=read(root/f"pairs-{task}.json")["ids"]
  visibility=read(root/"spair-visibility-audit.json") if task=="spair" else None
  source_excluded={row["pair_id"] for row in visibility["measurements"] if row["valid_keypoints"]==0} if visibility else set()
  if "first_source" in payload["protocol"]:
   protocol=payload["protocol"];first,second=read(protocol["first_source"]),read(protocol["second_source"])
   args.stage=protocol["first_stage"];args.second_stage=protocol["second_stage"]
   assert args.stage==key["first_stage"] and args.second_stage==key["second_stage"]
   names=tuple(protocol["difference"].split("_minus_",1))
   for original in (first,second):validate_correspondence(original,selected,visibility)
   pairing=validate_pairing(first,second,args.stage+":27",args.second_stage+":27")
   assert pairing["same_excluded_ids"] and {row["pair_id"] for row in pairing["exclusion_union"]}==source_excluded
   replay=correspondence(first,second,names,args)
   assert all(payload[key]==value for key,value in replay.items())
  else:
   assert payload["first"]==key["first_stage"]+":27" and payload["second"]==key["second_stage"]+":27"
   first=read(root/"measurements/correspondence"/f"kimi_k31_pro_correspondence_{task}.json")
   validate_correspondence(first,selected,visibility)
   pairing=validate_pairing(first,first,payload["first"],payload["second"])
   assert pairing["same_excluded_ids"] and {row["pair_id"] for row in pairing["exclusion_union"]}==source_excluded
   assert payload["first_advantage"]==interval(first["measurements"],payload["first"],payload["second"],PRIMARY[task],args)
   rows=first["measurements"]
   if task=="spair":
    bins={str(value):[row for row in rows if row["viewpoint"]==value] for value in (0,1,2)}
   else:
    edges=[0,15,30,60,180] if task=="scannet" else [0,30,60,90,120]
    bins={f"[{low},{high})":[row for row in rows if low<=row["angle"]<high] for low,high in zip(edges,edges[1:])}
   assert payload["by_difficulty"]=={name:interval(rows,payload["first"],payload["second"],PRIMARY[task],args) for name,rows in bins.items() if rows}
 else:raise AssertionError(task)
 results.append(dict(source=str(path),sha256=digest,task=task,required_comparison_cells=len(records),
  bootstrap_threads=torch.get_num_threads(),exact_numerical_replay=True,cells=[record["cell"] for record in records]))
 print(json.dumps(dict(task=task,source=path.name,cells=len(records),exact_replay=True)),flush=True)
assert sum(row["required_comparison_cells"] for row in results)==102
actual=[json.dumps(record["cell"],sort_keys=True) for record in book["completed_paired"]]
assert len(actual)==len(set(actual))==102 and set(actual)=={json.dumps(cell,sort_keys=True) for cell in expected["paired_comparison_cells"]}
report=dict(passed=True,comparison_cells=102,source_files=len(results),all_intervals_replayed_exactly=True,
 source_backed_spair_exclusions_verified=True,resamples=10000,seed=0,results=results,
 verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/"all-paired-intervals-independent-replay.json").write_text(json.dumps(report,indent=2)+"\n")
print("All102requiredcomparisoncells replayed exactly from actual rows",flush=True)
