"""Log one bash command. A GPU command holds its physical device flock."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--phase", required=True)
parser.add_argument("--gpu", choices=("0", "1"))
parser.add_argument("command")
args = parser.parse_args()
root = Path(__file__).parent
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=args.gpu or "", HF_HOME="/tmp/vtb-delivery/hf",
           OMP_NUM_THREADS="8" if args.gpu else "4", MKL_NUM_THREADS="8" if args.gpu else "4")
sources = sorted([*Path("vtb").rglob("*.py"), *Path("scripts").glob("*.py"), *Path("scripts").glob("*.sh"), Path("uv.lock")])
sources += sorted(root.glob("*.py"))
source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources if path.is_file()}
row = dict(phase=args.phase, command=args.command, cwd=os.getcwd(), gpu=args.gpu,
           threads=8 if args.gpu else 4, implementation_sha256=source_hashes,
           git_head=subprocess.check_output(["bash", "-ic", "git rev-parse HEAD"], env=env, stderr=subprocess.DEVNULL, text=True).strip(),
           git_dirty=subprocess.check_output(["bash", "-ic", "git status --porcelain"], env=env, stderr=subprocess.DEVNULL, text=True).splitlines(),
           started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           inventory_sha256=hashlib.sha256((root / "expected-inventory.json").read_bytes()).hexdigest())
if args.gpu:
    grant = root / "gpu-grant.json"
    if not grant.exists() or json.loads(grant.read_text()).get("physical_gpu") != int(args.gpu):
        raise SystemExit("No explicit reservation recorded in gpu-grant.json")
    command = ["flock", "-x", f"/tmp/vtb-delivery/locks/gpu{args.gpu}.lock", "bash", "-ic", args.command]
else:
    command = ["bash", "-ic", args.command]
start = time.monotonic()
with (root / "execution.jsonl").open("a") as out:
    out.write(json.dumps(row | {"event": "start"}) + "\n")
with (root / f"{args.phase}.log").open("w") as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
with (root / "execution.jsonl").open("a") as out:
    out.write(json.dumps(row | {"event": "end", "exit_code": result.returncode,
                              "elapsed_seconds": time.monotonic() - start,
                              "ended": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
raise SystemExit(result.returncode)
