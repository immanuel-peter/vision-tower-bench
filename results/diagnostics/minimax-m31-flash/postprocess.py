"""CPU paired statistics with explicit visibility exclusions and preserved sources."""
import argparse,hashlib,json,math,os,shlex,subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from vtb.correspondence import paired_bootstrap
from vtb.semantic_bootstrap import paired_image_bootstrap
from scripts.cross_run_bootstrap import class_macro_bootstrap
torch.set_num_threads(2)
root=Path("/tmp/vtb-delivery/issues/21");results=root/"measurements"
python=str(Path.cwd()/".venv/bin/python");flash="minimax_m31_flash";models=(flash,"minimax_m3")
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
args=SimpleNamespace(resamples=10000,confidence=.95,bootstrap_seed=0)
parser=argparse.ArgumentParser();parser.add_argument("--section",choices=("all","correspondence","remaining"),default="all");phase=parser.parse_args()
visibility=json.loads((root/"spair-visibility-audit.json").read_text())["source_visibility"]
source_zero={key for key,value in visibility.items() if value["visible_keypoints"]==0}
def canonical(path):
 relative=path.relative_to(results)
 if path.name.startswith("minimax_m3_"):
  return str(Path("results/diagnostics/minimax-m31-flash/m3-image-isolation")/relative)
 return str(Path("results")/relative)
def source(path):return {canonical(path):hashlib.sha256(path.read_bytes()).hexdigest()}
def run(command):
 subprocess.run(["bash","-ic",shlex.join([str(x) for x in command])],env=env,check=True)
if phase.section in ("all","correspondence"):
 for model in models:
  for dataset in ("navi","scannet","spair"):
   path=results/"correspondence"/f"{model}_correspondence_{dataset}.json"
   target=results/"bootstrap"/f"correspondence-{dataset}-{model}-projected-vs-tower.json"
   run([python,"scripts/correspondence_bootstrap.py","--run",path,"--out",target])
   payload=json.loads(path.read_text());out=json.loads(target.read_text());metric=out["protocol"]["metric"]
   finite=[];excluded=[]
   for row in payload["measurements"]:
    first=row["cells"]["projected:32"];second=row["cells"]["tower:32"]
    if all(math.isfinite(c[metric]) for c in (first,second)):
     finite.append({"pair_id":row["pair_id"],"projected":first[metric],"tower":second[metric],**({"class":row["class"]} if "class" in row else {})})
    else:
     if dataset=="spair":
      assert row["pair_id"] in source_zero
      assert first["keypoints"]==second["keypoints"]==visibility[row["pair_id"]]["visible_keypoints"]==0
      reason="source-verified zero visible keypoints in both images after common crop"
     else:
      assert first["matches"]==second["matches"]==0
      reason="zero valid geometric correspondences"
     excluded.append({"pair_id":row["pair_id"],"reason":reason})
   x=torch.tensor([r["projected"] for r in finite]);y=torch.tensor([r["tower"] for r in finite])
   expected=paired_bootstrap(x,y)
   for key in ("point_estimate","lower","upper"):assert abs(expected[key]-out["first_advantage"][key])<1e-7
   if dataset=="spair":
    assert {r["pair_id"] for r in excluded}==source_zero
    out["class_macro_first_advantage"]=class_macro_bootstrap(finite,("projected","tower"),args)
   out["sources"]=source(path)
   out["paired_selection"]={"selected_pairs":len(payload["measurements"]),"finite_intersection_ids":[r["pair_id"] for r in finite],"excluded_union":excluded}
   out["paired_point_estimates"]={"projected":x.mean().item(),"tower":y.mean().item()}
   target.write_text(json.dumps(out,indent=2)+"\n")
 for dataset in ("navi","scannet","spair"):
  first=results/"correspondence"/f"{flash}_correspondence_{dataset}.json"
  second=results/"correspondence"/f"minimax_m3_correspondence_{dataset}.json"
  for stage in ("tower","merged","projected"):
   target=results/"bootstrap"/f"correspondence-{dataset}-m31-vs-m3-{stage}.json"
   run([python,"scripts/cross_run_bootstrap.py","correspondence","--first",first,"--first-name",flash,"--second",second,"--second-name","minimax_m3","--stage",stage,"--out",target])
   out=json.loads(target.read_text())
   if dataset=="spair":
    assert set(out["excluded_nonfinite_pair_ids"])==source_zero
    out["excluded_union"]=[{"pair_id":key,"reason":"source-verified zero visible keypoints","visible_keypoints":0} for key in sorted(source_zero)]
    assert out["finite_pairs"]==len(visibility)-len(source_zero)
   out["sources"]=source(first)|source(second)
   out["protocol"]["stage"]=stage
   target.write_text(json.dumps(out,indent=2)+"\n")
if phase.section in ("all","remaining"):
 # Materialize identity evidence already logged as clean correctness for all transforms.
 identity_report={}
 for model in models:
  path=results/"perturbation"/f"{model}_perturbation.json"
  original=path.read_bytes();payload=json.loads(original)
  if "identity_evidence" in payload:
   identity_report[model]=payload["identity_evidence"]|{"augmented_sha256":hashlib.sha256(original).hexdigest()}
   continue
  backup=root/"raw-probe-payloads"/path.name;backup.parent.mkdir(exist_ok=True)
  if backup.exists():assert backup.read_bytes()==original
  else:backup.write_bytes(original)
  for cell in payload["cells"]:
   names={entry["condition"] for entry in cell["conditions"]}
   assert len(names)==13 and "identity" in names
   assert set(cell["measurements"])==names-{"identity"}
   reference=None
   for condition,rows in cell["measurements"].items():
    clean=[(row["image_id"],row["clean"]) for row in rows]
    if reference is None:reference=clean
    assert clean==reference,(model,cell["stage"],condition,"clean vectors differ")
   assert len(reference)==300 and len({key for key,_ in reference})==300
   assert all(len(values)==3 and all(x in (0,1) for x in values) for _,values in reference)
   values=torch.tensor([values for _,values in reference]).T
   identity=next(entry for entry in cell["conditions"] if entry["condition"]=="identity")
   assert abs(values.float().mean().item()-identity["accuracy"])<1e-7
   assert all(abs(x-y)<1e-7 for x,y in zip(values.float().mean(dim=1).tolist(),identity["accuracy_by_seed"]))
   cell["measurements"]["identity"]=[{"image_id":key,"clean":values,"perturbed":values} for key,values in reference]
  payload["identity_evidence"]={"derivation":"Lossless copy of same-run clean correctness, verified identical across all12 transformed conditions; no readout, score, seed or protocol change.","original_sha256":hashlib.sha256(original).hexdigest(),"original_runtime_payload":str(backup),"original_published_payload":"results/diagnostics/minimax-m31-flash/raw-probe-payloads/"+path.name,"source_runner":"scripts/perturbation_run.py","source_runner_sha256":hashlib.sha256(Path("scripts/perturbation_run.py").read_bytes()).hexdigest(),"materialization_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  path.write_text(json.dumps(payload,indent=2)+"\n")
  identity_report[model]=payload["identity_evidence"]|{"augmented_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
 (root/"identity-evidence-materialization.json").write_text(json.dumps(identity_report,indent=2)+"\n")
 first=results/"perturbation"/f"{flash}_perturbation.json";second=results/"perturbation"/"minimax_m3_perturbation.json"
 for stage in ("tower","merged","projected"):
  target=results/"bootstrap"/f"perturbation-m31-vs-m3-{stage}.json"
  run([python,"scripts/cross_run_bootstrap.py","perturbation","--first",first,"--first-name",flash,"--second",second,"--second-name","minimax_m3","--stage",stage,"--out",target])
  out=json.loads(target.read_text());out["sources"]=source(first)|source(second);out["protocol"]["stage"]=stage
  target.write_text(json.dumps(out,indent=2)+"\n")
 for model in models:
  path=results/"perturbation"/f"{model}_perturbation.json";payload=json.loads(path.read_text())
  cells={row["stage"]:row for row in payload["cells"]};intervals={}
  for condition,rows in cells["projected"]["measurements"].items():
   a={row["image_id"]:row for row in rows};b={row["image_id"]:row for row in cells["tower"]["measurements"][condition]}
   assert a.keys()==b.keys();ids=sorted(a)
   x=torch.tensor([a[i]["perturbed"] for i in ids]).T;y=torch.tensor([b[i]["perturbed"] for i in ids]).T
   intervals[condition]={"image_ids":ids,"projected_by_seed":x.tolist(),"tower_by_seed":y.tolist(),"difference":paired_image_bootstrap(x,y)}
  out={"sources":source(path),"first":"projected","second":"tower","conditions":intervals,"protocol":{"resamples":10000,"bootstrap_seed":0,"pairing":"same-run seed-mean per-image correctness"}}
  (results/"bootstrap"/f"perturbation-{model}.json").write_text(json.dumps(out,indent=2)+"\n")
  runs={}
  for path in sorted((results/"standardized").glob(f"{model}_probe_*.json")):
   value=json.loads(path.read_text());assert value["images"]==13000 and len(value["cells"])==1
   assert value["cells"][0]["reducer_diagnostic"]=="channel_standardization_sample_std_correction1"
   runs[path.stem]=value
  assert len(runs)==4
  (results/"standardized"/f"{model}_label_budget_standardized.json").write_text(json.dumps({"arm":"standardized","format":"one result per key","image_inference":"independent single-image native forwards","runs":runs},indent=2)+"\n")

