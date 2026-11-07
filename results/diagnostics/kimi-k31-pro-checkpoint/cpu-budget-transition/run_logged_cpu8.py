"""Log an explicitly authorized CPU8 phase after the prior CPU4 fit boundary."""

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
parser.add_argument("command")
args = parser.parse_args()
root = Path(__file__).parent
prior = [json.loads(line) for line in (root / "execution.jsonl").read_text().splitlines()
         if json.loads(line).get("phase") == "cpu-matched-recognition-resume"]
if not prior or prior[-1]["event"] != "end" or prior[-1]["exit_code"] != 0:
    raise SystemExit("Current CPU4 command must finish normally before CPU8 starts")
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HOME="/tmp/vtb-delivery/hf", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
paths = [*Path("vtb").rglob("*.py"), *Path("scripts").glob("*.py"), *Path("scripts").glob("*.sh"), *root.glob("*.py"), Path("uv.lock")]
row = dict(phase=args.phase, command=args.command, cwd=os.getcwd(), gpu=None, threads=8,
           thread_policy="CPU budget8; incomplete paired counterpart may retain4 to match an existing fit; per-cell actual threads recorded",
           implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()},
           git_head=subprocess.check_output(["bash", "-ic", "git rev-parse HEAD"], env=env, stderr=subprocess.DEVNULL, text=True).strip(),
           started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           inventory_sha256=hashlib.sha256((root / "expected-inventory.json").read_bytes()).hexdigest())
with (root / "execution.jsonl").open("a") as stream:
    stream.write(json.dumps(row | {"event": "start"}) + "\n")
began = time.monotonic()
with (root / f"{args.phase}.log").open("w") as log:
    result = subprocess.run(["bash", "-ic", args.command], env=env, stdout=log, stderr=subprocess.STDOUT)
with (root / "execution.jsonl").open("a") as stream:
    stream.write(json.dumps(row | {"event": "end", "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - began, "ended": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
raise SystemExit(result.returncode)
