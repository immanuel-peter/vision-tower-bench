"""Run CPU regressions sequentially after the original benchmark and audit jobs."""
import datetime
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

root = Path(__file__).parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "8"
for gpu in (0, 1):
    release = json.loads((root / f"scheduler/gpu{gpu}-release.json").read_text())
    assert release["passed"] and release["memory_mib"] == 0 and not release["cuda_processes"]
assert json.loads((root / "full-experiment-inventory-hash-audit.json").read_text())["passed"]
env = os.environ | {"VTB_SKIP_WEIGHTS": "1", "PYTHONPATH": "."}
command = ".venv/bin/python -m pytest -q"
paths = [p for folder in ("vtb", "scripts", "tests", "examples") for p in Path(folder).rglob("*.py")]
record = dict(phase="final-cpu8-regression", command=command, cuda_visible_devices="", threads=8,
              skip_weights=True, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
with (root / "execution.jsonl").open("a") as f:
    f.write(json.dumps(record | {"event": "start"}) + "\n")
began = time.monotonic()
log = root / "final-regression-tests.log"
with log.open("w") as out:
    result = subprocess.run(["bash", "-ic", command], env=env, stdout=out, stderr=subprocess.STDOUT)
output = log.read_text()
counts = {label: int(matches[-1]) if (matches := re.findall(r"(\d+) " + label, output)) else 0
          for label in ("passed", "skipped", "failed", "errors")}
record.update(passed=result.returncode == 0, exit_code=result.returncode,
              passed_tests=counts["passed"], skipped_tests=counts["skipped"],
              failed_tests=counts["failed"], error_tests=counts["errors"],
              elapsed_seconds=time.monotonic() - began,
              ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
(root / "final-regression-tests.json").write_text(json.dumps(record, indent=2) + "\n")
with (root / "execution.jsonl").open("a") as f:
    f.write(json.dumps(record | {"event": "end"}) + "\n")
print(output[-12000:])
raise SystemExit(result.returncode)
