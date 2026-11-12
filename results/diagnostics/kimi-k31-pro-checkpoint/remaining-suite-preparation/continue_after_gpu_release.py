"""Wait on the kernel file event, then run CPU pairs only after verified GPU release."""
import ctypes,datetime,hashlib,json,os,shlex,struct,subprocess
from pathlib import Path
root=Path(__file__).parent
release_path=root/"remaining-gpu-release.json"
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
assert os.environ["OMP_NUM_THREADS"]==os.environ["MKL_NUM_THREADS"]=="8"
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno(),"inotify_init1")
watch=libc.inotify_add_watch(fd,os.fsencode(root),0x8|0x80)
if watch<0:raise OSError(ctypes.get_errno(),"inotify_add_watch")
print("Waiting for verified GPU0 release by kernel file notification; no CPU compute or CUDA initialization",flush=True)
try:
 while not release_path.exists():
  events=os.read(fd,65536)
  offset=0
  while offset<len(events):
   _,_,_,length=struct.unpack_from("iIII",events,offset)
   offset+=16+length
finally:os.close(fd)
release=json.loads(release_path.read_text())
assert release["passed"] and release["physical_gpu"]==0 and release["memory_mib"]==0 and not release["cuda_processes"]
assert release["exclusive_lock_reacquired"] and release["exclusive_lock_released"]
grant=json.loads((root/"gpu-grant.json").read_text());assert grant["active"] is False and grant["physical_gpu"]==0
rows=[json.loads(line) for line in (root/"execution.jsonl").read_text().splitlines()]
prior=[row for row in rows if row.get("phase")=="gpu0-remaining-suite"]
assert prior[-1]["event"]=="end" and prior[-1]["exit_code"]==0
progress=json.loads((root/"remaining-gpu-progress.json").read_text())
assert progress["complete"] and progress["scope"]=="all" and len(progress["results"])==43
book=json.loads((root/"completed-inventory.json").read_text())
assert book["completed_primary_cells"]==286 and book["completed_paired_cells"]==94 and not book["pending"]
command="PYTHONPATH=. .venv/bin/python "+shlex.quote(str(root/"cpu_remaining_correspondence_pairs.py"))
argv=[".venv/bin/python",str(root/"run_logged_cpu8.py"),"--phase","cpu8-final-correspondence-pairs",command]
result=subprocess.run(["bash","-ic",shlex.join(argv)])
if result.returncode:raise SystemExit(result.returncode)
book=json.loads((root/"completed-inventory.json").read_text())
expected=json.loads((root/"expected-inventory.json").read_text())
verified=[]
for collection,expected_key,count in (("completed","cells",286),("completed_paired","paired_comparison_cells",102)):
 keys=[json.dumps(record["cell"],sort_keys=True) for record in book[collection]]
 assert len(keys)==len(set(keys))==count and set(keys)=={json.dumps(cell,sort_keys=True) for cell in expected[expected_key]}
 for record in book[collection]:
  source=Path(record["source"])
  assert hashlib.sha256(source.read_bytes()).hexdigest()==record["sha256"]
  verified.append(dict(collection=collection,cell=record["cell"],source=str(source),sha256=record["sha256"]))
report=dict(passed=True,primary_cells=286,paired_cells=102,all_expected_cell_ids_exact=True,all_actual_source_hashes_verified=True,
 source_records=verified,gpu_release_before_cpu_tail=release,final_figures_docs_tests_and_inference_gate_still_required=True,
 completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/"full-experiment-inventory-hash-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print("All286primary/102paired experiment entries match exact inventory IDs and actual source hashes; final reporting/tests and published inference gate remain required",flush=True)
