"""Execute only an actual newly published example under a separately granted GPU1 lock."""
import csv,datetime,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence-dir",type=Path,required=True)
args=parser.parse_args()
root=args.evidence_dir
fresh=json.loads((root/"fresh-download.json").read_text());snapshot=Path(fresh["snapshot"])
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1"
assert os.environ["OMP_NUM_THREADS"]==os.environ["MKL_NUM_THREADS"]=="2"
assert (snapshot/"examples/inference.py").is_file(),"Published example missing; never substitute bench code"
for relative,digest in fresh["files_sha256"].items():
 assert hashlib.sha256((snapshot/relative).read_bytes()).hexdigest()==digest
old=json.loads(Path("/tmp/vtb-delivery/issues/24/inference-smoke/fresh-download.json").read_text())
for name in ("model.safetensors","projector.safetensors"):
 assert fresh["files_sha256"][name]==old["files_sha256"][name],"Published weights changed; existing conversion evidence cannot silently transfer"
def query(command):return subprocess.check_output(["bash","-ic",command],text=True,stderr=subprocess.DEVNULL).strip()
uuid,name,memory=next(csv.reader([query("nvidia-smi -i 1 --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits")]))
uuid,name,memory=uuid.strip(),name.strip(),int(memory.strip())
processes=[line for line in query("nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader").splitlines() if uuid in line]
assert memory==0 and not processes,"Verify idle before CUDA initialization"
image=Path(json.loads(Path("/tmp/vtb-delivery/issues/24/inference-smoke/candidate-cuda-smoke.json").read_text())["image"])
python=Path.cwd()/".venv/bin/python"
argv=[str(python),"examples/inference.py",str(image)]
started_at=datetime.datetime.now(datetime.timezone.utc).isoformat();began=time.monotonic()
with (root/"published-example.stdout.json").open("w") as out,(root/"published-example.stderr.txt").open("w") as err:
 process=subprocess.run(["bash","-ic",shlex.join(argv)],cwd=snapshot,stdout=out,stderr=err)
report=dict(passed=False,repo=fresh["repo"],revision=fresh["revision"],snapshot=str(snapshot),command=shlex.join(argv),cwd=str(snapshot),
 exit_code=process.returncode,physical_gpu=1,gpu_uuid=uuid,gpu_name=name,memory_mib_before_cuda=memory,threads=2,
 image=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),published_example_sha256=fresh["files_sha256"]["examples/inference.py"],
 started_at=started_at,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_seconds=time.monotonic()-began,
 actual_published_script_executed=True,bench_script_substituted=False,benchmark_weight_hashes_unchanged=True,
 baseline_benchmark_release=old["revision"],scope="functional published-example gate; production official TORCH_SDPA BF16 parity remains failed")
if process.returncode==0:
 actual=json.loads((root/"published-example.stdout.json").read_text())
 assert actual["model_dir"]==str(snapshot.absolute())
 assert actual["batch_size"]==1 and actual["device"]=="cuda:0" and actual["dtype"]=="torch.bfloat16" and actual["torch_threads"]==2
 assert actual["image_sha256"]==report["image_sha256"]
 for stage,shape in {"tower":[1,1024,1024],"merged":[1,256,4096],"projected":[1,256,7168]}.items():
  assert actual["stages"][stage]==dict(shape=shape,device="cuda:0",dtype="torch.bfloat16")
 assert all(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()==digest for name,digest in fresh["files_sha256"].items())
 report.update(passed=True,actual_inference=actual,fresh_snapshot_unchanged_after_smoke=True)
(root/"published-example-gate.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({key:value for key,value in report.items() if key!="actual_inference"}))
raise SystemExit(process.returncode)
