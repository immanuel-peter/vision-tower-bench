"""Execute the downloaded card command after a driver-only idle check."""
import csv,datetime,hashlib,json,os,re,shlex,subprocess,time
from pathlib import Path
r=Path("/tmp/vtb-delivery/issues/21/hf-followup");publication=json.loads((r/"publication.json").read_text());snapshot=Path(publication["snapshot"])
def query(name,args):
 p=subprocess.run(["bash","-ic",shlex.join(["nvidia-smi","--id=0"]+args)],check=True,capture_output=True,text=True)
 (r/name).write_text(p.stdout);return list(csv.reader(p.stdout.splitlines()))
before=query("gpu-before.csv",["--query-gpu=index,uuid,memory.used","--format=csv,noheader,nounits"])
processes=query("processes-before.csv",["--query-compute-apps=pid,process_name,gpu_uuid","--format=csv,noheader,nounits"])
assert len(before)==1 and before[0][0].strip()=="0" and int(before[0][2].strip())==0,before
uuid=before[0][1].strip();assert not [row for row in processes if row and row[-1].strip()==uuid]
# Torch is first imported in the child, after the idle driver queries above.
from PIL import Image
image=Path("/tmp/vtb-delivery/data/imagenet100/validation/validation_000000.jpg")
with Image.open(image) as img:size=list(img.size)
card=(snapshot/"README.md").read_text();blocks=re.findall(r"```bash\n(.*?)```",card,re.S);assert len(blocks)==1
command=blocks[0].strip().replace("/path/to/image.jpg",shlex.quote(str(image)))
assert command=="python examples/inference.py "+str(image)
activation="source "+shlex.quote(str(Path.cwd()/".venv/bin/activate"))+"\n"+command
started=datetime.datetime.now(datetime.timezone.utc).isoformat();tic=time.monotonic()
with (r/"cuda-smoke.log").open("w") as log:result=subprocess.run(["bash","-ic",activation],cwd=snapshot,stdout=log,stderr=subprocess.STDOUT)
text=(r/"cuda-smoke.log").read_text();shapes={"tower":[1,1024,1280],"merged":[1,256,5120],"projected":[1,256,6144]}
passed=result.returncode==0 and all(f"{name} {tuple(shape)} dtype torch.bfloat16 device cuda:0" in text for name,shape in shapes.items()) and "bundle "+str(snapshot) in text
post=query("gpu-after.csv",["--query-gpu=index,uuid,memory.used","--format=csv,noheader,nounits"])
post_processes=query("processes-after.csv",["--query-compute-apps=pid,process_name,gpu_uuid","--format=csv,noheader,nounits"])
assert int(post[0][2].strip())==0 and not [row for row in post_processes if row and row[-1].strip()==uuid]
report={"passed":passed,"exit_code":result.returncode,"repo":publication["repo"],"tested_revision":publication["revision"],"snapshot":str(snapshot),"snapshot_symlink_script":(snapshot/"examples/inference.py").is_symlink(),"command":command,"card_command_template":blocks[0].strip(),"inline_python_snippets":[],"device":"cuda:0","physical_gpu":0,"CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],"dtype":"torch.bfloat16","batch_size":1,"cpu_threads":2,"image":str(image),"image_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),"original_image_size":size,"crop_size":[448,448],"shapes":shapes,"started_utc":started,"elapsed_seconds":time.monotonic()-tic,"driver_idle_verified_before_torch_import":True,"driver_memory_mib_after":0,"active_compute_processes_after":[],"lock":"/tmp/vtb-delivery/locks/gpu0.lock"}
(r/"cuda-smoke.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
if not passed:raise SystemExit("Downloaded card inference failed; see cuda-smoke.log")
