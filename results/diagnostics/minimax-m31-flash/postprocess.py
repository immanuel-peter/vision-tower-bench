"""CPU paired postprocessing after the issue-21 measurement inventory is complete."""
import hashlib,json,os,shlex,subprocess
from pathlib import Path
import torch
from vtb.semantic_bootstrap import paired_image_bootstrap
root=Path("/tmp/vtb-delivery/issues/21")
results=root/"measurements"
python=str(Path.cwd()/".venv/bin/python")
flash="minimax_m31_flash"
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=""
def run(args):
 subprocess.run(["bash","-ic",shlex.join([str(x) for x in args])],env=env,check=True)
for dataset in ("navi","scannet","spair"):
 first=results/"correspondence"/f"{flash}_correspondence_{dataset}.json"
 second=results/"correspondence"/f"minimax_m3_correspondence_{dataset}.json"
 run([python,"scripts/correspondence_bootstrap.py","--run",first,"--out",results/"bootstrap"/f"correspondence-{dataset}-{flash}.json"])
 for stage in ("tower","merged","projected"):
  run([python,"scripts/cross_run_bootstrap.py","correspondence","--first",first,"--first-name",flash,"--second",second,"--second-name","minimax_m3","--stage",stage,"--out",results/"bootstrap"/f"correspondence-{dataset}-m31-vs-m3-{stage}.json"])
first=results/"perturbation"/f"{flash}_perturbation.json"
second=results/"perturbation"/"minimax_m3_perturbation.json"
for stage in ("tower","merged","projected"):
 run([python,"scripts/cross_run_bootstrap.py","perturbation","--first",first,"--first-name",flash,"--second",second,"--second-name","minimax_m3","--stage",stage,"--out",results/"bootstrap"/f"perturbation-m31-vs-m3-{stage}.json"])
payload=json.loads(first.read_text())
cells={row["stage"]:row for row in payload["cells"]}
intervals={}
for condition,rows in cells["projected"]["measurements"].items():
 a={row["image_id"]:row for row in rows}
 b={row["image_id"]:row for row in cells["tower"]["measurements"][condition]}
 assert a.keys()==b.keys()
 ids=sorted(a)
 x=torch.tensor([a[i]["perturbed"] for i in ids]).T
 y=torch.tensor([b[i]["perturbed"] for i in ids]).T
 intervals[condition]={"image_ids":ids,"projected_by_seed":x.tolist(),"tower_by_seed":y.tolist(),"difference":paired_image_bootstrap(x,y,resamples=10000,confidence=.95,seed=0)}
out={"source":str(first),"source_sha256":hashlib.sha256(first.read_bytes()).hexdigest(),"first":"projected","second":"tower","conditions":intervals,"protocol":{"resamples":10000,"bootstrap_seed":0,"pairing":"same-run seed-mean per-image correctness"}}
(results/"bootstrap"/f"perturbation-{flash}.json").write_text(json.dumps(out,indent=2)+"\n")
runs={}
for path in sorted((results/"standardized").glob("*.json")):
 value=json.loads(path.read_text())
 assert value["images"]==13000 and len(value["cells"])==1
 assert value["cells"][0]["reducer_diagnostic"]=="channel_standardization_sample_std_correction1"
 runs[path.stem]=value
assert len(runs)==4
(results/"standardized"/"label_budget_standardized.json").write_text(json.dumps({"arm":"standardized","format":"one result per key","runs":runs},indent=2)+"\n")

