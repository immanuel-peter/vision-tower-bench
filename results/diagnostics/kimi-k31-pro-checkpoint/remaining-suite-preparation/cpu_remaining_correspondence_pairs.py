"""Validate the actual ScanNet/SPair finite intersection and run eight CPU8 pairs."""
import hashlib,json,math,os,shlex,shutil,subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.cross_run_bootstrap import correspondence
from scripts.correspondence_bootstrap import interval,PRIMARY
from scripts.validate_kimi_results import validate_correspondence,validate_pairing,finite_numbers

root=Path("/tmp/vtb-delivery/issues/24")
public=Path("results/diagnostics/kimi-k31-pro-checkpoint")
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="" and torch.get_num_threads()==8
plan=json.loads((root/"fullsuite-plan.json").read_text())
expected=json.loads((root/"expected-inventory.json").read_text())
book=json.loads((root/"completed-inventory.json").read_text())
results=[]
args=SimpleNamespace(metric=None,resamples=10000,confidence=.95,bootstrap_seed=0)
for dataset in ("scannet","spair"):
 ids=json.loads((root/f"pairs-{dataset}.json").read_text())["ids"]
 visibility=json.loads((root/"spair-visibility-audit.json").read_text()) if dataset=="spair" else None
 source_excluded={row["pair_id"] for row in visibility["measurements"] if row["valid_keypoints"]==0} if visibility else set()
 jobs=[job for job in plan["cpu_post_jobs"] if job["kind"]=="correspondence" and f"_{dataset}.json" in job["first"]]
 assert len(jobs)==4
 for job in jobs:
  first,second=[json.loads(Path(job[key]).read_text()) for key in ("first","second")]
  for payload in (first,second):
   audit=validate_correspondence(payload,ids,visibility)
   assert {row["pair_id"] for row in audit["exclusions"]}==source_excluded
  first_runtime=json.loads(Path(job["first"]).with_suffix(".runtime.json").read_text())["adapters"][0]
  second_runtime=json.loads(Path(job["second"]).with_suffix(".runtime.json").read_text())["adapters"][0]
  first_proof=json.loads(Path(job["first"]).with_suffix(".validation.json").read_text())
  second_proof=json.loads(Path(job["second"]).with_suffix(".validation.json").read_text())
  assert first_proof["gpu_identity"]==second_proof["gpu_identity"] and first_proof["physical_gpu"]==second_proof["physical_gpu"]
  for field in ("device","dtype","cuda_visible_devices","threads","native_integrity"):
   assert first_runtime[field]==second_runtime[field]
  assert first_runtime["device"]=="cuda:0" and first_runtime["threads"]==8
  first_key,second_key=f"{job['stage']}:27",f"{job['second_stage']}:27"
  audit=validate_pairing(first,second,first_key,second_key)
  assert audit["same_excluded_ids"] and {row["pair_id"] for row in audit["exclusion_union"]}==source_excluded
  assert audit["finite_intersection"]==len(ids)-len(source_excluded)
  out=Path(job["out"])
  resumed=out.exists()
  if not resumed:
   with out.with_suffix(".log").open("w") as log:
    process=subprocess.run(["bash","-ic",shlex.join(job["argv"])],stdout=log,stderr=subprocess.STDOUT)
   if process.returncode:raise SystemExit(process.returncode)
  payload=json.loads(out.read_text())
  assert finite_numbers(payload)
  sibling=job["first"]!=job["second"]
  args.stage,args.second_stage=job["stage"],job["second_stage"]
  if sibling:
   replay=correspondence(first,second,(job["first_name"],job["second_name"]),args)
   assert all(payload[key]==value for key,value in replay.items())
   assert set(payload["excluded_nonfinite_pair_ids"])==source_excluded
   assert payload["finite_pairs"]==audit["finite_intersection"]
   assert {row["pair_id"] for row in payload["measurements"]}==set(audit["finite_ids"])
   summary=payload["difference"]
  else:
   replay=interval(first["measurements"],first_key,second_key,PRIMARY[dataset],args)
   assert payload["first_advantage"]==replay and replay["pairs"]==audit["finite_intersection"]
   summary=payload["first_advantage"]
  left={row["pair_id"]:row for row in first["measurements"]}
  right={row["pair_id"]:row for row in second["measurements"]}
  point=sum(left[pair]["cells"][first_key][PRIMARY[dataset]]-right[pair]["cells"][second_key][PRIMARY[dataset]] for pair in audit["finite_ids"])/len(audit["finite_ids"])
  assert abs(summary["point_estimate"]-point)<1e-6
  key=dict(comparison="kimi_k31_pro-minus-moonvit_v2" if sibling else "kimi_k31_pro-projected-minus-tower",task=dataset,first_stage=job["stage"],second_stage=job["second_stage"])
  assert key in expected["paired_comparison_cells"]
  record=dict(cell=key,source=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),pairing_audit=audit,
   actual_point_from_same_rows=point,exact_overall_bootstrap_replay=True,exact_class_macro_replay=sibling and dataset=="spair",
   bootstrap_device="cpu",bootstrap_threads=8,feature_and_match_device="cuda:0",physical_gpu=first_runtime["cuda_visible_devices"],
   source_hashes={job[name]:hashlib.sha256(Path(job[name]).read_bytes()).hexdigest() for name in ("first","second")},resumed=resumed)
  old=[row for row in book["completed_paired"] if row["cell"]==key]
  if old:assert len(old)==1 and old[0]["sha256"]==record["sha256"]
  else:book["completed_paired"].append(record)
  out.with_suffix(".validation.json").write_text(json.dumps(record,indent=2)+"\n")
  for source in (out,out.with_suffix(".validation.json")):shutil.copy2(source,Path("results/bootstrap")/source.name)
  results.append(record)
  keys={json.dumps(row["cell"],sort_keys=True) for row in book["completed_paired"]}
  assert len(keys)==len(book["completed_paired"]) and keys<={json.dumps(row,sort_keys=True) for row in expected["paired_comparison_cells"]}
  book["completed_paired_cells"]=len(keys)
  for path in (root/"completed-inventory.json",public/"completed-inventory.json"):path.write_text(json.dumps(book,indent=2)+"\n")
  job.update(status="complete_content_verified_cpu",result_sha256=record["sha256"])
  (root/"cpu-remaining-correspondence-pairs-progress.json").write_text(json.dumps(dict(complete=False,results=results),indent=2)+"\n")
  print(json.dumps(dict(dataset=dataset,comparison=key,finite_pairs=audit["finite_intersection"],source_zero_keypoint_pairs=len(source_excluded),passed=True)),flush=True)
assert len(results)==8
(root/"cpu-remaining-correspondence-pairs-progress.json").write_text(json.dumps(dict(complete=True,results=results),indent=2)+"\n")
(root/"fullsuite-plan.json").write_text(json.dumps(plan,indent=2)+"\n")
