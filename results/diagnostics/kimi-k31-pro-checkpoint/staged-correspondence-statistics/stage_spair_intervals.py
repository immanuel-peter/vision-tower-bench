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
for dataset in ("spair",):
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

  record=dict(passed=True,source=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),finite_pairs=audit["finite_intersection"],point_from_actual_rows=point,exact_bootstrap_replay=True,threads=8,device="cpu",feature_and_match_physical_gpu=first_proof["physical_gpu"],source_sha256={job[name]:hashlib.sha256(Path(job[name]).read_bytes()).hexdigest() for name in ("first","second")},no_model_training=True,inventory_registration="deferred to existing post-GPU CPU continuation; output will be reused")
  out.with_suffix(".staged-validation.json").write_text(json.dumps(record,indent=2)+"\n")
  results.append(record)
  print(json.dumps(record),flush=True)
assert len(results)==4
(root/"staged-spair-intervals.json").write_text(json.dumps(dict(passed=True,comparisons=4,results=results,no_inventory_mutation=True,no_cuda=True,no_training=True),indent=2)+"\n")
