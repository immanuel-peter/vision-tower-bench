"""Verify an empty granted device under its lock, then record and revoke the grant."""
import argparse,datetime,fcntl,json,subprocess
from pathlib import Path
root=Path("/tmp/vtb-delivery/issues/24")
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpu",type=int,choices=(0,1),required=True)
args=parser.parse_args()
grant=json.loads((root/"gpu-grant.json").read_text())
assert grant["physical_gpu"]==args.gpu and grant.get("active") is True
assert grant["phase"].startswith("remaining-")
with Path(f"/tmp/vtb-delivery/locks/gpu{args.gpu}.lock").open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 command=f"nvidia-smi -i {args.gpu} --query-gpu=uuid,memory.used --format=csv,noheader,nounits"
 device=subprocess.check_output(["bash","-ic",command],text=True).strip()
 uuid,memory=[value.strip() for value in device.split(",")]
 processes=subprocess.check_output(["bash","-ic","nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader"],text=True).strip().splitlines()
 active=[line for line in processes if uuid in line]
 assert int(memory)==0 and not active,(device,active)
 report=dict(passed=True,physical_gpu=args.gpu,gpu_uuid=uuid,memory_mib=int(memory),cuda_processes=active,
  exclusive_lock_reacquired=True,grant=grant,verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 grant.update(active=False,released_at=report["verified_at"],release_evidence="remaining-gpu-release.json")
 (root/"gpu-grant.json").write_text(json.dumps(grant,indent=2)+"\n")
 (root/"remaining-gpu-release-under-lock.json").write_text(json.dumps(report,indent=2)+"\n")
 fcntl.flock(lock,fcntl.LOCK_UN)
report["exclusive_lock_released"]=True
(root/"remaining-gpu-release.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
