"""Collect only content-validated complete matrices into the repository's standard layout."""
import importlib.util,json,shutil,hashlib
from pathlib import Path
E=Path("/tmp/vtb-delivery/issues/22")
ROOT=Path("/home/shadeform/.t3/worktrees/vision-tower-bench/t3-b75c6e2e")
R=ROOT/"results";M="qwen38_flash_next"
spec=importlib.util.spec_from_file_location("suite",E/"suite.py");suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite);suite.build()
evidence=R/"evidence/qwen38-flash-next"
valid={};errors={}
for job in suite.jobs:
    if not Path(job["output"]).exists():continue
    try:
        if job["kind"] not in ["cache","perturbation-cache"]:
            valid[job["name"]]=suite.validate(job)
    except Exception as error:errors[job["name"]]=repr(error)
for task,readouts,arms in [("recognition",["attention","mean"],["raw","matched"]),("geometry",["depth","normal"],["raw","matched"]),("kitti",["depth"],["matched"])]:
 for readout in readouts:
  for arm in arms:
   wanted=suite.SLICES if task!="kitti" else [("tower",27),("projected",27)]
   names=[f"{M}_{readout}_{arm}_{stage}_L{layer:02d}" for stage,layer in wanted]
   if not all(("kitti-"+n if task=="kitti" else n) in valid for n in names):continue
   payloads=[json.loads((E/"measurements"/task/"cells"/(n+".json")).read_text()) for n in names]
   payload=payloads[0]
   payload["cells"]=[cell for p in payloads for cell in p["cells"]]
   assert len(payload["cells"])==len(wanted)
   name=f"{M}_probe_{readout}_{arm}.json" if task=="recognition" else f"{M}_geometry_{readout}_{arm}.json"
   (R/task/name).write_text(json.dumps(payload,indent=2)+"\n")
mapping={}
for job in suite.jobs:
 if job["name"] not in valid or job["kind"] in ["cache","perturbation-cache"]:continue
 source=Path(job["output"]);relative=source.relative_to(E/"measurements")
 if "cells" in relative.parts:continue
 target=(evidence/"controls"/source.name) if source.name.startswith("qwen3_5_") else R/relative
 if relative.parts[0]=="bootstrap":
  stem=source.stem
  if stem==f"recognition-{M}-projector-projected":
   stem=f"semantic-{M}-projected-vs-{M}-tower"
  elif stem==f"kitti-{M}-projector-projected":
   stem=f"geometry-depth-kitti-matched-{M}-projected-vs-tower"
  elif stem==f"perturbation-{M}-projected-vs-tower":
   stem=f"perturbation-{M}-projector-comparison"
  elif stem.startswith(f"recognition-{M}-sibling-"):
   stem=stem.replace(f"recognition-{M}-sibling-",f"semantic-{M}-vs-qwen3_5-")
  target=R/"bootstrap"/(stem+".json")
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(source,target);mapping[str(source)]=str(target.relative_to(ROOT))
# Keep copied cross-run source references reviewable in a fresh clone.
for source,target in mapping.items():
 if "/qwen38-flash-next-conditioning/" not in target:continue
 path=ROOT/target;payload=json.loads(path.read_text());provenance=payload.get("provenance",{})
 if provenance.get("primary_source") in mapping:
  provenance["primary_source"]=mapping[provenance["primary_source"]]
 path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
for source,target in mapping.items():
 if "/bootstrap/" not in "/"+target and not Path(target).name.startswith("paired-"):continue
 path=ROOT/target;payload=json.loads(path.read_text());protocol=payload.get("protocol",{})
 hashes={}
 for key in ["first_source","second_source","source"]:
  if key in protocol and protocol[key] in mapping:
   rewritten=mapping[protocol[key]];protocol[key]=rewritten
   hashes[rewritten]=hashlib.sha256((ROOT/rewritten).read_bytes()).hexdigest()
 if hashes:payload["sources"]=hashes
 path.write_text(json.dumps(payload,indent=2)+"\n")
# Extend the standardized diagnostic source without replacing any existing run.
diag=R/"diagnostics/mimo26-tower-outlier-channel/label_budget_standardized.json"
payload=json.loads(diag.read_text())
for path in (R/"diagnostics/standardized-label-budget").glob(M+"_*.json"):
 payload["runs"][path.stem]=json.loads(path.read_text())
diag.write_text(json.dumps(payload,indent=2)+"\n")
report={"planned_jobs":len(suite.jobs),"validated_measurement_jobs":len(valid),"errors":errors,
        "missing_measurement_jobs":[j["name"] for j in suite.jobs if j["kind"] not in ["cache","perturbation-cache"] and j["name"] not in valid],
        "standard_recognition_cells":sum(len(json.loads(p.read_text())["cells"]) for p in (R/"recognition").glob(M+"_probe_*.json")),
        "standard_geometry_cells":sum(len(json.loads(p.read_text())["cells"]) for p in (R/"geometry").glob(M+"_geometry_*.json"))}
(E/"content-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

