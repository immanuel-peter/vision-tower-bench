"""Private one-shot process-admission guard; never imports a model or Torch."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
policy_path=Path(os.environ.get("VTB_ADMISSION_POLICY","/tmp/vtb-delivery/issues/24/scheduler/admission-policy.json"))
if policy_path.is_file():
 policy=json.loads(policy_path.read_text())
 args=sys.orig_argv
 if policy.get("enabled") and os.environ.get("CUDA_VISIBLE_DEVICES")==policy["visible_device"] and "--out" in args:
  target=args[args.index("--out")+1]
  if target in policy["targets"]:
   proof=dict(exit_code=75,kind="planned unstarted-process admission handoff, not a failed fit",target=target,argv=args,
    torch_not_imported="torch" not in sys.modules,cuda_model_never_started=True,pid=os.getpid(),
    hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
   destination=Path(policy["proof"])
   with destination.open("w") as stream:
    json.dump(proof,stream,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
   os.write(1,b"Planned process boundary: current matrix already completed; next fit not started; handoff exit75\n")
   os._exit(75)
