"""Audit exact pending-inventory coverage without importing Torch or initializing CUDA."""
import hashlib,json,runpy
from collections import Counter
from pathlib import Path
root=Path("/tmp/vtb-delivery/issues/24")
namespace=runpy.run_path(str(root/"gpu_remaining_suite.py"),run_name="remaining_plan_audit")
queue=namespace["jobs"]()
expected=json.loads((root/"expected-inventory.json").read_text())
book=json.loads((root/"completed-inventory.json").read_text())
keys=[]
for row in queue:
 if row["kind"]=="geometry":
  keys += [dict(task=row["task"],model=row["model"],stage=stage,layer=layer,arm=row["arm"]) for stage,layer in
   [("tower",n) for n in (3,7,10,14,17,20,24,27)]+[("merged",27),("projected",27)]]
 elif row["kind"]=="kitti":
  keys += [dict(task="kitti",model=row["model"],stage=stage,layer=27,arm="matched") for stage in ("tower","projected")]
 elif row["kind"]=="correspondence":
  keys += [dict(task=row["dataset"],model=row["model"],stage=stage,layer=27) for stage in ("tower","merged","projected")]
 elif row["kind"]=="throughput":keys += [dict(task="throughput",model=row["model"])]
canonical=lambda values:{json.dumps(value,sort_keys=True) for value in values}
actual=canonical(keys)
assert len(actual)==len(keys)==98
remaining=[cell for cell in book["pending"] if cell["task"]!="recognition"]
assert actual==canonical(remaining)
paired=[expected["paired_comparison_cells"][row["paired_index"]] for row in queue if row["kind"]=="geometry-pair"]
assert len(canonical(paired))==len(paired)==27
completed_paired=canonical([row["cell"] for row in book["completed_paired"]])
pending=canonical(expected["paired_comparison_cells"])-completed_paired
assert canonical(paired)<=pending
uncovered=[json.loads(value) for value in pending-canonical(paired)]
assert len(uncovered)==8 and {value["task"] for value in uncovered}=={"scannet","spair"}
report=dict(passed=True,cuda_initialized=False,compute_launched=False,prepared_jobs=len(queue),
 primary_cells_covered=98,primary_task_counts=dict(Counter(row["task"] for row in keys)),
 paired_dense_cells_covered=27,cpu_correspondence_comparisons_afterward=uncovered,
 raw_recognition_cpu_phase_separate=True,driver_sha256=hashlib.sha256((root/"gpu_remaining_suite.py").read_bytes()).hexdigest(),
 expected_inventory_sha256=hashlib.sha256((root/"expected-inventory.json").read_bytes()).hexdigest())
(root/"remaining-gpu-plan-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({key:value for key,value in report.items() if key!="cpu_correspondence_comparisons_afterward"}))
