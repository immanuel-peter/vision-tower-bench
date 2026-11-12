"""Run an auditable CPU-only delivery phase and retain its full log."""
import argparse,datetime,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("command",nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==["--"] else a.command
assert command
r=Path("/tmp/vtb-delivery/issues/21");log_path=r/f"{a.name}.log"
if log_path.exists():
 previous=log_path.with_name(log_path.stem+"."+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".previous.log");previous.write_bytes(log_path.read_bytes())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",HF_HOME="/tmp/vtb-delivery/hf",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",MPLCONFIGDIR=str(r/"matplotlib"),PYTHONPATH=str(Path.cwd()))
command=["env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=2","MKL_NUM_THREADS=2"]+command
paths=[f for directory in ("vtb","scripts","tests") for f in Path(directory).rglob("*.py")]+list(r.glob("*.py"))+[Path("uv.lock")]
hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
start=datetime.datetime.now(datetime.timezone.utc).isoformat();tic=time.monotonic()
with log_path.open("w") as log:
 rc=subprocess.run(["bash","-ic",shlex.join(command)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
record={"name":a.name,"command":shlex.join(command),"start_utc":start,"end_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-tic,"exit_code":rc,"source_hashes":hashes,"resources":{"device":"cpu","CUDA_VISIBLE_DEVICES":"","threads":2},"log":str(log_path)}
with (r/"execution.jsonl").open("a") as log:log.write(json.dumps(record)+"\n")
print(a.name,"rc",rc,"elapsed",round(record["elapsed_seconds"],3),"log",log_path,flush=True)
if rc:raise SystemExit(rc)
