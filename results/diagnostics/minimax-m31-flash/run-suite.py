"""Serial issue-21 commands. Launch only while issue 21 owns GPU 0."""
import argparse,datetime,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path("/tmp/vtb-delivery/issues/21")
DATA=Path("/tmp/vtb-delivery/data")
PY=str(Path.cwd() / ".venv/bin/python")
FEATURES=ROOT/"features"
RESULTS=ROOT/"measurements"
for d in ("recognition","geometry","label-budget","correspondence","perturbation","kitti","bootstrap","throughput"):
 (RESULTS/d).mkdir(parents=True,exist_ok=True)
jobs=[]
def add(category,name,args,ready=()):
 jobs.append((category,name,[str(x) for x in args],ready))
flash="minimax_m31_flash"
models=(flash,"minimax_m3")
for model in models:
 add("extract",f"extract-recognition-{model}",[PY,"-m","vtb.extract","--model",model,"--images",DATA/"imagenet100/validation","--out",FEATURES/"recognition","--pool",4,"--batch-size",8,"--workers",2,"--device","cuda:0"],("imagenet100",))
 for dataset in ("diode","kitti"):
  args=[PY,"-m","vtb.extract","--model",model,"--images",DATA/dataset/"val","--out",FEATURES/dataset,"--batch-size",8,"--workers",2,"--device","cuda:0"]
  if dataset=="kitti": args+=["--final-stages","tower","projected"]
  add("extract",f"extract-{dataset}-{model}",args,(dataset,))
 run=FEATURES/"recognition"/f"{model}_448_pool4"
 add("recognition",f"recognition-{model}",["bash","-ic",f"GPUS=0 LANES=1 THREADS=8 bash scripts/semantic_matrix.sh {FEATURES}/recognition {DATA}/imagenet100/validation_labels.json {RESULTS}/recognition {model}_448_pool4"])
 add("label-budget",f"label-budget-{model}",["bash","-ic",f"GPUS=0 LANES=1 THREADS=8 bash scripts/label_budget_matrix.sh {FEATURES}/recognition {DATA}/imagenet100/validation_labels.json {RESULTS}/label-budget {model}_448_pool4"])
 add("geometry",f"geometry-{model}",["bash","-ic",f"GPUS=0 LANES=1 THREADS=8 bash scripts/geometry_matrix.sh {FEATURES}/diode {DATA}/diode/val_targets.npz {RESULTS}/geometry {model}_448_full"],("diode",))
 add("kitti",f"kitti-{model}",[PY,"-m","vtb.geometry_run","--run",FEATURES/"kitti"/f"{model}_448_full","--targets",DATA/"kitti/val_targets.npz","--task","depth","--match-capacity","--stages","tower","projected","--deepest-only","--device","cuda:0","--out",RESULTS/"kitti"/f"{model}_geometry_depth_matched.json"],("kitti",))
 add("throughput",f"throughput-{model}",[PY,"scripts/tower_throughput.py","--model",model,"--images",DATA/"imagenet100/validation","--batch-size",8,"--workers",2,"--device","cuda:0","--out",RESULTS/"throughput"/f"{model}.json"],("imagenet100",))
 for dataset in ("spair","scannet","navi"):
  roots={"spair":"spair/SPair-71k","scannet":"scannet/scannet_test_1500","navi":"navi/navi_v1"}
  add("correspondence",f"correspondence-{dataset}-{model}",[PY,"scripts/correspondence_run.py","--model",model,"--dataset",dataset,"--root",DATA/"correspondence"/roots[dataset],"--resolution",448,"--out",RESULTS/"correspondence"/f"{model}_correspondence_{dataset}.json"],("correspondence",))
 add("perturbation",f"perturbation-{model}",["bash","-ic",f"MODELS={model} GPUS=0 LANES=1 THREADS=8 WORKERS=2 bash scripts/perturbation_matrix.sh {DATA}/imagenet100/validation {DATA}/perturbation {FEATURES}/perturbation {DATA}/imagenet100/validation_labels.json {DATA}/perturbation/transforms.manifest.json {RESULTS}/perturbation"],("perturbation",))
for dataset,tasks in (("diode",("depth","normal")),("kitti",("depth",))):
 for task in tasks:
  for arm in ("matched","raw") if dataset=="diode" else ("matched",):
   common=[PY,"scripts/geometry_bootstrap.py","--first-run",FEATURES/dataset/f"{flash}_448_full","--first-name",flash,"--first-layer",32,"--second-layer",32,"--targets",DATA/dataset/"val_targets.npz","--task",task,"--arm",arm,"--device","cuda:0"]
   add("bootstrap",f"bootstrap-{dataset}-{task}-{arm}-projector",common+["--first-name",flash+"_projected","--first-stage","projected","--second-run",FEATURES/dataset/f"{flash}_448_full","--second-name",flash+"_tower","--second-stage","tower","--out",RESULTS/"bootstrap"/f"geometry-{task}-{'kitti-' if dataset=='kitti' else ''}{arm}-{flash}-projected-vs-tower.json"],(dataset,))
   for stage in ("tower","merged","projected") if dataset=="diode" else ("tower","projected"):
    add("sibling",f"sibling-{dataset}-{task}-{arm}-{stage}",common+["--first-stage",stage,"--second-run",FEATURES/dataset/"minimax_m3_448_full","--second-name","minimax_m3","--second-stage",stage,"--out",RESULTS/"bootstrap"/f"geometry-{task}-{dataset}-{arm}-m31-vs-m3-{stage}.json"],(dataset,))
for stage in ("tower","merged","projected"):
 args=[PY,"scripts/semantic_bootstrap.py","--first-run",FEATURES/"recognition"/f"{flash}_448_pool4","--first-name",flash,"--first-layer",32,"--first-stage",stage,"--second-run",FEATURES/"recognition"/"minimax_m3_448_pool4","--second-name","minimax_m3","--second-layer",32,"--second-stage",stage,"--labels",DATA/"imagenet100/validation_labels.json","--out",RESULTS/"bootstrap"/f"semantic-m31-vs-m3-{stage}.json"]
 add("sibling",f"sibling-recognition-{stage}",args,("imagenet100",))
add("bootstrap","bootstrap-recognition-projector",[PY,"scripts/semantic_bootstrap.py","--first-run",FEATURES/"recognition"/f"{flash}_448_pool4","--first-name",flash+"_projected","--first-layer",32,"--first-stage","projected","--second-run",FEATURES/"recognition"/f"{flash}_448_pool4","--second-name",flash+"_tower","--second-layer",32,"--second-stage","tower","--labels",DATA/"imagenet100/validation_labels.json","--out",RESULTS/"bootstrap"/f"semantic-attention-matched-{flash}-projected-minus-tower.json"],("imagenet100",))
for model in models:
 for fraction in (0.01,0.05,0.2,1.0):
  tag="f"+str(fraction).replace(".","")
  add("standardized",f"standardized-{model}-{tag}",[PY,"-m","vtb.probe_run","--run",FEATURES/"recognition"/f"{model}_448_pool4","--labels",DATA/"imagenet100/validation_labels.json","--readout","attention","--match-capacity","--standardize-channels","--stages","tower","--depth-points",32,"--label-fraction",fraction,"--device","cuda:0","--out",RESULTS/"standardized"/f"{model}_probe_attention_matched_{tag}.json"],("imagenet100",))
(RESULTS/"standardized").mkdir(parents=True,exist_ok=True)
for dataset,tasks in (("diode",("depth","normal")),("kitti",("depth",))):
 for task in tasks:
  for arm in ("matched","raw") if dataset=="diode" else ("matched",):
   model="minimax_m3"
   add("control-bootstrap",f"control-bootstrap-{dataset}-{task}-{arm}",[PY,"scripts/geometry_bootstrap.py","--first-run",FEATURES/dataset/f"{model}_448_full","--first-name",model+"_projected","--first-stage","projected","--first-layer",32,"--second-run",FEATURES/dataset/f"{model}_448_full","--second-name",model+"_tower","--second-stage","tower","--second-layer",32,"--targets",DATA/dataset/"val_targets.npz","--task",task,"--arm",arm,"--device","cuda:0","--out",RESULTS/"bootstrap"/f"geometry-{task}-{'kitti-' if dataset=='kitti' else ''}{arm}-{model}-projected-vs-tower.json"],(dataset,))
model="minimax_m3"
add("control-bootstrap","control-bootstrap-recognition",[PY,"scripts/semantic_bootstrap.py","--first-run",FEATURES/"recognition"/f"{model}_448_pool4","--first-name",model+"_projected","--first-stage","projected","--first-layer",32,"--second-run",FEATURES/"recognition"/f"{model}_448_pool4","--second-name",model+"_tower","--second-stage","tower","--second-layer",32,"--labels",DATA/"imagenet100/validation_labels.json","--out",RESULTS/"bootstrap"/f"semantic-attention-matched-{model}-projected-minus-tower.json"],("imagenet100",))
p=argparse.ArgumentParser()
p.add_argument("--category",required=True)
p.add_argument("--only")
p.add_argument("--list",action="store_true")
a=p.parse_args()
for category,name,args,ready in jobs:
 if category!=a.category or a.only and a.only not in name:continue
 if a.only and a.only.endswith("minimax_m3") and not name.endswith("minimax_m3"):continue
 if a.list: print(name,shlex.join(args));continue
 for marker in ready:
  assert (DATA/f"{marker}.READY").is_file(),f"{marker} not ready"
 command=shlex.join(args)
 env=os.environ.copy()
 env.update(HF_HOME="/tmp/vtb-delivery/hf",CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",GPUS="0",LANES="1",THREADS="8",WORKERS="2",VTB_MINIMAX_M3_BUNDLE=str(ROOT/"m3-bundle"),VTB_GEOMETRY_PAIRED_CACHE=str(ROOT/"paired-cell-cache"))
 hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for base in ("vtb","scripts") for f in Path(base).rglob("*.py")}
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();tic=time.monotonic()
 log_path = ROOT/f"{name}.log"
 if log_path.exists():
  backup = log_path.with_name(log_path.stem+"."+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".previous.log")
  backup.write_bytes(log_path.read_bytes())
 with log_path.open("w") as log:
  rc=subprocess.run(["flock","/tmp/vtb-delivery/locks/gpu0.lock","bash","-ic",command],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 if rc == 0:
  model = flash if flash in name else "minimax_m3"
  expected = []
  if category == "recognition":
   expected = [(RESULTS/"recognition"/f"{model}_probe_{readout}_{arm}.json",10) for readout in ("attention","mean") for arm in ("raw","matched")]
  elif category == "geometry":
   expected = [(RESULTS/"geometry"/f"{model}_geometry_{task}_{arm}.json",10) for task in ("depth","normal") for arm in ("raw","matched")]
  elif category == "label-budget":
   expected = [(RESULTS/"label-budget"/f"{model}_probe_{readout}_matched_{fraction}.json",1) for readout in ("attention","mean") for fraction in ("f001","f005","f020","f10")]
  elif category == "perturbation":
   expected = [(RESULTS/"perturbation"/f"{model}_perturbation.json",3)]
  for output,count in expected:
   try:
    payload=json.loads(output.read_text())
    assert len(payload["cells"])==count
    if category=="perturbation":assert all(len(cell["conditions"])==13 for cell in payload["cells"])
   except Exception as exc:
    print(f"Output validation failed for {output}: {exc}",flush=True);rc=90
 record={"name":name,"command":command,"start_utc":start,"elapsed_seconds":time.monotonic()-tic,"exit_code":rc,"source_hashes":hashes,"elapsed_includes_gpu_lock_wait":True,"resources":{"physical_gpu":0,"logical_device":"cuda:0","threads":8,"loader_workers_max":2,"lock":"/tmp/vtb-delivery/locks/gpu0.lock"}}
 with (ROOT/"execution.jsonl").open("a") as log:log.write(json.dumps(record)+"\n")
 if rc:raise SystemExit(f"{name} failed: {rc}")
 print(name,"done",flush=True)

