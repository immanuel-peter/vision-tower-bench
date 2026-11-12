"""Audit complete, separately recorded GPU sibling cache proofs after extraction."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path("/tmp/vtb-delivery/issues/24")
phase=json.loads((root/"gpu-extraction-phase-progress.json").read_text())
assert phase["complete"] is True and len(phase["reports"])==30
models={model:{} for model in ("kimi_k31_pro","moonvit_v2")}
for row in phase["reports"]:
 run=Path(row["run"])
 model=run.name.split("_448_")[0]
 progress=json.loads((run/"progress.json").read_text())
 assert progress["complete"] is True
 report=json.loads((run/"cache-validation.json").read_text())
 assert report["passed"] is True
 proof=hashlib.sha256((run/"cache-validation.json").read_bytes()).hexdigest()
 assert proof==progress["content_validation_sha256"]==row["content_validation_sha256"]
 provenance=json.loads((run/"provenance.json").read_text())
 assert provenance==progress["provenance"]==row["provenance"]
 kind,condition=provenance["kind"],provenance["condition"]
 expected={"semantic":(13000,10,510),"kitti":(1000,2,8),"perturbation":(2000,3,24)}[kind]
 assert (report["images"],report["slices"],report["shards"])==expected
 assert row["images"]==expected[0] and row["slices"]==expected[1]
 assert len(report["files"])==expected[2]
 ids_name={"semantic":"imagenet100","kitti":"kitti","perturbation":"perturbation"}[kind]
 ids=json.loads((root/f"ids-{ids_name}.json").read_text())["ids"]
 assert provenance["ordered_ids_sha256"]==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 assert provenance["device"]=="cuda:0" and provenance["physical_gpu"]==1
 assert provenance["batch_size"]==1 and provenance["threads"]==8 and provenance["workers"]==2
 models[model][(kind,condition)]=dict(run=str(run),provenance=provenance,images=report["images"],slices=report["slices"],shards=report["shards"],bytes=row["bytes"],seconds=row["seconds"],validation_sha256=proof)
assert models["kimi_k31_pro"].keys()==models["moonvit_v2"].keys()
pairs=[]
ignore={"model_id","release_revision","parent_repo","parent_revision"}
for key in models["kimi_k31_pro"]:
 a,b=(models[model][key] for model in models)
 assert {k:v for k,v in a["provenance"].items() if k not in ignore}=={k:v for k,v in b["provenance"].items() if k not in ignore}
 pairs.append(dict(kind=key[0],condition=key[1],first=a,second=b,same_device_settings=True))
out=dict(passed=True,complete=True,updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),pairs=pairs,
         gpu_images=sum(row["images"] for row in phase["reports"]),gpu_shards=sum(row["shards"] for row in phase["reports"]),
         bytes=sum(row["bytes"] for row in phase["reports"]),seconds=phase["seconds"],
         cpu_diode_reused_separately="cpu-geometry-pair.json",quality_metric_cells_completed=0)
(root/"gpu-extraction-pair-audit.json").write_text(json.dumps(out,indent=2)+"\n")
plan=json.loads((root/"fullsuite-plan.json").read_text())
for job in plan["jobs"]:
 if job["phase"]!="extraction" or job.get("status")=="satisfied_by_verified_cpu_cache":
  continue
 for model in models:
  for kind in ("semantic","kitti","perturbation"):
   if job["name"]==f"{model}-{kind}":
    completed=[row for (task,_),row in models[model].items() if task==kind]
    job["status"]="complete_content_verified_gpu_cache"
    job["completed_caches"]=[row["run"] for row in completed]
    job["content_validation_sha256"]=[row["validation_sha256"] for row in completed]
plan["remaining_extraction_complete"]=True
(root/"fullsuite-plan.json").write_text(json.dumps(plan,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="pairs"},indent=2))
