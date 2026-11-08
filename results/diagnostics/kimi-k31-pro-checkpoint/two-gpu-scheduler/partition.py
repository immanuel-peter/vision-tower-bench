"""Resume unique unstarted jobs after a documented pre-model process handoff."""
import argparse,ctypes,datetime,fcntl,hashlib,json,os,runpy,shlex,shutil,struct,subprocess,time
from pathlib import Path
ROOT=Path("/tmp/vtb-delivery/issues/24")
STATE=ROOT/"scheduler"
PUBLIC=Path("results/diagnostics/kimi-k31-pro-checkpoint")
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+f".tmp-{os.getpid()}")
 temporary.write_text(json.dumps(value,indent=2)+"\n");os.replace(temporary,path)
def command(argv,**kwargs):return subprocess.run(["bash","-ic",shlex.join(argv)],**kwargs)
def journal(row):
 with (STATE/"locks/journal.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with (ROOT/"execution.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
def hardware(gpu):
 import csv
 def query(text):return subprocess.check_output(["bash","-ic",text],text=True,stderr=subprocess.DEVNULL).strip()
 uuid,name,memory=next(csv.reader([query(f"nvidia-smi -i {gpu} --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits")]))
 uuid,name,memory=uuid.strip(),name.strip(),int(memory.strip())
 processes=[line for line in query("nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader").splitlines() if uuid in line]
 return dict(physical_gpu=gpu,gpu_uuid=uuid,gpu_name=name,memory_mib=memory,cuda_processes=processes)
def register(row,report):
 path=Path(row["out"]);payload=read(path);expected=read(ROOT/"expected-inventory.json")
 with (STATE/"locks/inventory.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  book=read(ROOT/"completed-inventory.json")
  if row["kind"]=="geometry-pair":
   keys=[expected["paired_comparison_cells"][row["paired_index"]]];collection="completed_paired";destination=Path("results/bootstrap")
  else:
   collection="completed";model=row["model"]
   if row["kind"] in {"geometry","kitti"}:
    keys=[dict(task=row["task"] if row["kind"]=="geometry" else "kitti",model=model,stage=cell["stage"],layer=cell["layer_index"],arm=row["arm"]) for cell in payload["cells"]]
   elif row["kind"]=="correspondence":keys=[dict(task=row["dataset"],model=model,stage=cell["stage"],layer=cell["layer_index"]) for cell in payload["cells"]]
   else:keys=[dict(task="throughput",model=model)]
   folder="correspondence" if row["kind"]=="correspondence" else row["kind"]
   destination=Path("results")/folder if model=="kimi_k31_pro" else PUBLIC/"sibling-control"/folder
  destination.mkdir(parents=True,exist_ok=True)
  for key in keys:
   existing=[record for record in book[collection] if record["cell"]==key]
   if existing:assert len(existing)==1 and existing[0]["sha256"]==sha(path)
   else:book[collection].append(dict(cell=key,source=str(path),sha256=sha(path)))
  for collection,expected_key,count_key in (("completed","cells","completed_primary_cells"),("completed_paired","paired_comparison_cells","completed_paired_cells")):
   actual={json.dumps(record["cell"],sort_keys=True) for record in book[collection]}
   assert len(actual)==len(book[collection]) and actual<={json.dumps(cell,sort_keys=True) for cell in expected[expected_key]}
   book[count_key]=len(actual)
  actual={json.dumps(record["cell"],sort_keys=True) for record in book["completed"]}
  book["pending"]=[cell for cell in expected["cells"] if json.dumps(cell,sort_keys=True) not in actual]
  for source in (path,path.with_suffix(".validation.json"),path.with_suffix(".runtime.json")):
   if source.exists():shutil.copy2(source,destination/source.name)
  if path.with_suffix(".log").exists():shutil.copy2(path.with_suffix(".log"),destination/path.with_suffix(".txt").name)
  atomic(ROOT/"completed-inventory.json",book);atomic(PUBLIC/"completed-inventory.json",book)
def worker(gpu):
 assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu)
 assert os.environ["OMP_NUM_THREADS"]==os.environ["MKL_NUM_THREADS"]=="8"
 plan=read(STATE/"plan.json");assert plan["status"]=="active_after_verified_process_boundary"
 api=runpy.run_path(str(ROOT/"gpu_remaining_suite.py"),run_name="partition_api")
 phase=f"gpu{gpu}-partition-original-suite"
 journal(dict(event="start",phase=phase,gpu=gpu,threads=8,started=stamp(),plan_sha256=sha(STATE/"plan.json"),driver_sha256=sha(__file__)))
 with Path(f"/tmp/vtb-delivery/locks/gpu{gpu}.lock").open("a") as device_lock:
  print(f"Waiting for exclusive GPU{gpu} lock",flush=True);fcntl.flock(device_lock,fcntl.LOCK_EX)
  before=hardware(gpu);assert before["memory_mib"]==0 and not before["cuda_processes"],before
  for row in plan["jobs"]:
   if row["assigned_physical_gpu"]!=gpu:continue
   status_path=STATE/"jobs"/(row["job_id"]+".json")
   with (STATE/"locks"/(row["job_id"]+".lock")).open("a") as job_lock:
    fcntl.flock(job_lock,fcntl.LOCK_EX)
    status=read(status_path)
    assert status["job"]["assigned_physical_gpu"]==gpu
    if status["state"]=="complete":
     assert sha(row["out"])==status["validation"]["payload_sha256"]
     continue
    assert status["state"]=="pending","A running/failed claim requires explicit audit before resuming"
    out=Path(row["out"]);assert not out.exists(),f"Unexpected unclaimed output: {out}"
    status.update(state="running",pid=os.getpid(),started=stamp(),physical_gpu=gpu,gpu_identity=before)
    atomic(status_path,status)
    job_event=dict(phase=phase,job_id=row["job_id"],gpu=gpu,command=shlex.join(row["argv"]),started=status["started"],plan_sha256=sha(STATE/"plan.json"),driver_sha256=sha(__file__))
    journal(job_event|dict(event="job_start"))
    api["source_preflight"](row);out.parent.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    with out.with_suffix(".log").open("w") as log:
     process=command(row["argv"],env=os.environ|{"PYTHONPATH":".","GPUS":str(gpu),"LANES":"1","THREADS":"8"},stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:
     status.update(state="failed",exit_code=process.returncode,ended=stamp());atomic(status_path,status)
     journal(job_event|dict(event="job_end",exit_code=process.returncode,ended=stamp()))
     raise SystemExit(process.returncode)
    report=api["validate"](row)
    report.update(command=shlex.join(row["argv"]),elapsed_seconds=time.monotonic()-began,resumed=False,actual_device="cuda:0",physical_gpu=gpu,
     gpu_identity=before["gpu_uuid"]+", "+before["gpu_name"],threads=8,validated_at=stamp(),payload_sha256=sha(out),driver_sha256=sha(__file__),
     api_sha256=sha(ROOT/"gpu_remaining_suite.py"),assignment_plan_sha256=sha(STATE/"plan.json"),job_id=row["job_id"],source_snapshot_sha256=sha(STATE/"source-snapshot.json"))
    atomic(out.with_suffix(".validation.json"),report);register(row,report)
    status.update(state="complete",validation=report,ended=stamp(),exit_code=0);atomic(status_path,status)
    journal(job_event|dict(event="job_end",exit_code=0,ended=status["ended"],payload_sha256=report["payload_sha256"]))
    print(json.dumps(dict(job_id=row["job_id"],gpu=gpu,out=str(out),passed=True)),flush=True)
  fcntl.flock(device_lock,fcntl.LOCK_UN)
 with Path(f"/tmp/vtb-delivery/locks/gpu{gpu}.lock").open("a") as device_lock:
  fcntl.flock(device_lock,fcntl.LOCK_EX)
  after=hardware(gpu);assert after["memory_mib"]==0 and not after["cuda_processes"],after
  after.update(passed=True,exclusive_lock_reacquired=True,released_at=stamp())
  fcntl.flock(device_lock,fcntl.LOCK_UN)
 after["exclusive_lock_released"]=True;atomic(STATE/f"gpu{gpu}-release.json",after)
 if gpu==1:
  grant=read(STATE/"gpu1-benchmark-grant.json");grant.update(active=False,released_at=after["released_at"],release_evidence="scheduler/gpu1-release.json");atomic(STATE/"gpu1-benchmark-grant.json",grant)
 if gpu==0:
  grant=read(ROOT/"gpu-grant.json");grant.update(active=False,released_at=after["released_at"],release_evidence=f"scheduler/gpu{gpu}-release.json");atomic(ROOT/"gpu-grant.json",grant)
 journal(dict(event="end",phase=phase,gpu=gpu,threads=8,exit_code=0,ended=stamp(),release=after))
 print(json.dumps(after),flush=True)
def wait_legacy_boundary():
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
 assert fd>=0
 assert libc.inotify_add_watch(fd,os.fsencode(ROOT),0x8|0x80)>=0
 def ready():
  if not (STATE/"admission-boundary.json").exists():return False
  events=[json.loads(line) for line in (ROOT/"execution.jsonl").read_text().splitlines()]
  legacy=[row for row in events if row.get("phase")=="gpu0-remaining-suite"]
  return legacy[-1].get("event")=="end" and legacy[-1].get("exit_code")==75
 print("Waiting for current fit to finish normally and the next fresh process to defer before Torch/CUDA",flush=True)
 try:
  while not ready():os.read(fd,65536)
 finally:os.close(fd)
def coordinate():
 assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
 wait_legacy_boundary()
 boundary=read(STATE/"admission-boundary.json");assert boundary["exit_code"]==75 and boundary["torch_not_imported"]
 hook=Path("sitecustomize.py");assert sha(hook)==read(STATE/"admission-policy.json")["hook_sha256"]
 shutil.copy2(hook,STATE/"installed-sitecustomize.py")
 deferred_log=Path(boundary["target"]).with_suffix(".log")
 if deferred_log.exists():shutil.copy2(deferred_log,STATE/"legacy-deferred-process.txt")
 hook.unlink()
 for path in Path("__pycache__").glob("sitecustomize.*.pyc"):path.unlink()
 plan=read(STATE/"plan.json");old=read(ROOT/"remaining-gpu-progress.json")
 completed={row["job"]["out"]:row for row in old["results"]}
 for prior in old["results"]:
  original=prior["job"]
  for entry in plan["jobs"]:
   same=entry["out"]==original["out"]
   if original["kind"]=="geometry":same=entry["kind"]=="geometry" and entry["task"]==original["task"] and entry["arm"]==original["arm"]
   elif original["kind"]=="correspondence":same=entry["kind"]=="correspondence" and entry["dataset"]==original["dataset"]
   elif original["kind"]=="kitti":same=entry["kind"]=="kitti"
   if same:entry["assigned_physical_gpu"]=0
 for row in plan["jobs"]:
  status=dict(job=row,state="pending")
  if row["out"] in completed:
   prior=completed[row["out"]];assert prior["validation"]["physical_gpu"]==row["assigned_physical_gpu"]==0
   assert sha(row["out"])==prior["validation"]["payload_sha256"]
   status.update(state="complete",validation=prior["validation"],verified_legacy_reuse=True)
  atomic(STATE/"jobs"/(row["job_id"]+".json"),status)
 atomic(STATE/"source-snapshot.json",{str(path):sha(path) for path in [*Path("vtb").rglob("*.py"),*Path("scripts").glob("*.py"),Path(__file__),ROOT/"gpu_remaining_suite.py",STATE/"installed-sitecustomize.py"]})
 plan.update(status="active_after_verified_process_boundary",activated_at=stamp(),boundary=boundary,legacy_completed_jobs=len(completed),source_snapshot_sha256=sha(STATE/"source-snapshot.json"))
 atomic(STATE/"plan.json",plan)
 processes=[]
 for gpu in (0,1):
  env=os.environ|{"CUDA_VISIBLE_DEVICES":str(gpu),"OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8","THREADS":"8","GPUS":str(gpu),"LANES":"1","HF_HOME":"/tmp/vtb-delivery/hf"}
  log=(STATE/f"gpu{gpu}.log").open("w")
  argv=[str(Path.cwd()/".venv/bin/python"),str(Path(__file__).absolute()),"worker","--gpu",str(gpu)]
  processes.append((subprocess.Popen(["bash","-ic",shlex.join(argv)],env=env,stdout=log,stderr=subprocess.STDOUT),log))
 codes=[]
 for process,log in processes:codes.append(process.wait());log.close()
 if any(codes):raise SystemExit(next(code for code in codes if code))
 results=[]
 for row in plan["jobs"]:
  status=read(STATE/"jobs"/(row["job_id"]+".json"));assert status["state"]=="complete"
  assert sha(row["out"])==status["validation"]["payload_sha256"]
  results.append(dict(job=row,validation=status["validation"]))
 assert len(results)==43
 book=read(ROOT/"completed-inventory.json");assert book["completed_primary_cells"]==286 and book["completed_paired_cells"]==94
 atomic(ROOT/"remaining-gpu-progress.json",dict(complete=True,scope="all",results=results,partition_plan_sha256=sha(STATE/"plan.json")))
 zero,one=[read(STATE/f"gpu{gpu}-release.json") for gpu in (0,1)]
 assert all(value["passed"] and value["memory_mib"]==0 and not value["cuda_processes"] and value["exclusive_lock_released"] for value in (zero,one))
 journal(dict(event="end",phase="gpu0-remaining-suite",exit_code=0,ended=stamp(),event_kind="coordinated_completion_after_planned_handoff",legacy_handoff_exit75_preserved=True,
  boundary=boundary,partition_plan_sha256=sha(STATE/"plan.json"),completed_gpu_jobs=43,completed_primary_cells=286,completed_paired_cells=94,partitions=["gpu0-partition-original-suite","gpu1-partition-original-suite"]))
 atomic(ROOT/"remaining-gpu-release.json",zero|dict(both_original_suite_gpus_released=True,gpu1_release=one,coordinated_at=stamp()))
 print("Both GPU partitions completed and released; CPU continuation may proceed",flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("worker","coordinate"));parser.add_argument("--gpu",type=int,choices=(0,1));args=parser.parse_args()
 if args.mode=="worker":worker(args.gpu)
 else:coordinate()
