"""Verify GPU1 is empty after extraction, then release the reservation record."""

import datetime
import fcntl
import json
import os
import subprocess
from pathlib import Path

root = Path("/tmp/vtb-delivery/issues/24")
phase = json.loads((root / "gpu-extraction-phase-progress.json").read_text())
if not phase["complete"]:
    raise SystemExit("complete extraction required before normal phase release")
with Path("/tmp/vtb-delivery/locks/gpu1.lock").open("a") as stream:
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = subprocess.check_output(["bash", "-ic", "nvidia-smi --id=1 --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits"], text=True).strip()
    index, uuid, memory = [part.strip() for part in device.split(",")]
    raw = subprocess.check_output(["bash", "-ic", "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits"], text=True)
    processes = [line for line in raw.splitlines() if line.split(",")[0].strip() == uuid]
    if index != "1" or processes or float(memory) != 0:
        raise SystemExit(f"GPU1 release verification failed: {device}; {processes}")
    checked = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
record = dict(released=True, physical_gpu=1, gpu_uuid=uuid, used_memory_mib=0,
              cuda_processes=[], exclusive_lock_reacquired=True, lock_released=True,
              checked=checked, probe_pid=os.getpid(), phase="remaining-feature-extraction",
              next_scope="CPU-only four threads; no further CUDA without new explicit grant")
(root / "gpu-extraction-release.json").write_text(json.dumps(record, indent=2) + "\n")
(root / "gpu-grant.json").rename(root / "gpu-extraction-grant-consumed.json")
print(json.dumps(record, indent=2))
