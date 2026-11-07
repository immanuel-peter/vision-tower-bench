"""Complete corrected, image-independent MiniMax primary and paired inventories."""
import os,shlex,subprocess,sys
from pathlib import Path
root=Path("/tmp/vtb-delivery/issues/21");python=str(Path.cwd()/".venv/bin/python")
phases=[
 ("extract",None),("correspondence",None),
 ("bootstrap","diode-depth-matched"),("bootstrap","diode-normal-matched"),
 ("recognition",None),("standardized",None),("label-budget",None),
 ("geometry",None),("kitti",None),("throughput",None),("perturbation",None),
 ("sibling",None),("bootstrap","diode-depth-raw"),("bootstrap","diode-normal-raw"),
 ("bootstrap","kitti"),("bootstrap","recognition"),("control-bootstrap",None),
]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8")
for category,only in phases:
 args=[python,str(root/"run-suite.py"),"--category",category]
 if only:args+=["--only",only]
 print("START",category,only,flush=True)
 rc=subprocess.run(["bash","-ic",shlex.join(args)],env=env).returncode
 print("END",category,only,"rc",rc,flush=True)
 if rc:sys.exit(rc)
(root/"gpu-phases-complete").write_text("All corrected image-independent GPU phases returned success. Actual-content inventory validation remains mandatory.\n")

