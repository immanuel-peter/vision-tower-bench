import csv,datetime,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
root=Path(__file__).parent
repo_root=Path.cwd()
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1"
assert os.environ["OMP_NUM_THREADS"]==os.environ["MKL_NUM_THREADS"]=="2"
def command(text):
 return subprocess.check_output(["bash","-ic",text],text=True,stderr=subprocess.DEVNULL).strip()
uuid,name,memory=next(csv.reader([command("nvidia-smi -i 1 --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits")]))
uuid,name,memory=uuid.strip(),name.strip(),int(memory.strip())
processes=command("nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader").splitlines()
assert memory==0 and not [line for line in processes if uuid in line],"GPU1 must be empty before any CUDA initialization"
fresh=json.loads((root/"fresh-download.json").read_text());snapshot=Path(fresh["snapshot"])
assert fresh["revision"] in str(snapshot) and not (snapshot/"examples/inference.py").exists()
for relative,digest in fresh["files_sha256"].items():
 assert hashlib.sha256((snapshot/relative).read_bytes()).hexdigest()==digest
image=next(path for path in sorted(Path("/tmp/vtb-delivery/data/imagenet100/validation").rglob("*")) if path.suffix.lower() in {".jpg",".jpeg",".png"})
python=repo_root/".venv/bin/python"
published_argv=[str(python),"examples/inference.py",str(image)]
with (root/"published-example.stdout.txt").open("w") as out,(root/"published-example.stderr.txt").open("w") as err:
 published=subprocess.run(["bash","-ic",shlex.join(published_argv)],cwd=snapshot,stdout=out,stderr=err)
assert published.returncode!=0
blocked=dict(passed=False,publication_gate="blocked",reason="examples/inference.py absent from immutable published repository",repo=fresh["repo"],revision=fresh["revision"],snapshot=str(snapshot),command=shlex.join(published_argv),cwd=str(snapshot),exit_code=published.returncode,
 missing_files=["examples/inference.py"],gpu1_idle_before_cuda=True,physical_gpu=1,gpu_uuid=uuid)
(root/"published-example-gate.json").write_text(json.dumps(blocked,indent=2)+"\n")
argv=[str(python),str(repo_root/"examples/kimi_k31_pro_inference.py"),str(image),"--model-dir",str(snapshot),"--device","cuda:0","--resolution","448"]
started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
start=time.monotonic()
with (root/"candidate.stdout.json").open("w") as out,(root/"candidate.stderr.txt").open("w") as err:
 result=subprocess.run(["bash","-ic",shlex.join(argv)],stdout=out,stderr=err)
report=dict(passed=result.returncode==0,scope="bench-side proposed example against clean published snapshot; NOT the missing published-example gate",command=shlex.join(argv),cwd=str(repo_root),exit_code=result.returncode,
 revision=fresh["revision"],repo=fresh["repo"],snapshot=str(snapshot),physical_gpu=1,gpu_uuid=uuid,gpu_name=name,memory_mib_before_cuda=memory,torch_threads=2,
 image=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((repo_root/"examples/kimi_k31_pro_inference.py").read_bytes()).hexdigest(),
 started_at=started_at,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_seconds=time.monotonic()-start,published_example_gate_passed=False)
if result.returncode==0:
 actual=json.loads((root/"candidate.stdout.json").read_text())
 assert actual["batch_size"]==1 and actual["device"]=="cuda:0" and actual["dtype"]=="torch.bfloat16" and actual["torch_threads"]==2
 for stage,shape in {"tower":[1,1024,1024],"merged":[1,256,4096],"projected":[1,256,7168]}.items():
  assert actual["stages"][stage]==dict(shape=shape,device="cuda:0",dtype="torch.bfloat16")
 report["actual_inference"]=actual
(root/"candidate-cuda-smoke.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({key:value for key,value in report.items() if key!="actual_inference"}))
raise SystemExit(result.returncode)
