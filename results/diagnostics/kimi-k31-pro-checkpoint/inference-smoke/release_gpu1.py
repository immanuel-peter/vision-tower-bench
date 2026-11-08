import csv,datetime,fcntl,json,subprocess
from pathlib import Path
root=Path(__file__).parent
with Path("/tmp/vtb-delivery/locks/gpu1.lock").open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 def run(command):return subprocess.check_output(["bash","-ic",command],text=True,stderr=subprocess.DEVNULL).strip()
 uuid,memory=next(csv.reader([run("nvidia-smi -i 1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits")]))
 uuid,memory=uuid.strip(),int(memory.strip())
 own=[line for line in run("nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader").splitlines() if uuid in line]
 assert memory==0 and not own,(memory,own)
 report=dict(passed=True,physical_gpu=1,gpu_uuid=uuid,memory_mib=memory,cuda_processes=own,exclusive_lock_reacquired=True,verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 fcntl.flock(lock,fcntl.LOCK_UN)
report["exclusive_lock_released"]=True
(root/"gpu1-release.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
