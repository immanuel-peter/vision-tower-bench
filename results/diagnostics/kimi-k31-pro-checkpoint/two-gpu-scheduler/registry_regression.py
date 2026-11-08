import hashlib,json,multiprocessing,os,runpy,shutil,tempfile
from pathlib import Path
root=Path(__file__).parent
module=runpy.run_path(str(root/"partition.py"),run_name="registry_test")
register=module["register"];glob=register.__globals__
original=Path.cwd()
with tempfile.TemporaryDirectory(dir=root,prefix="registry-test-") as folder:
 test=Path(folder);glob["ROOT"]=test;glob["STATE"]=test/"scheduler";glob["PUBLIC"]=test/"public"
 (test/"scheduler/locks").mkdir(parents=True);(test/"measurements").mkdir()
 keys=[];jobs=[]
 for model in ("kimi_k31_pro","moonvit_v2"):
  source=Path("/tmp/vtb-delivery/issues/24/measurements/geometry")/f"{model}_geometry_depth_raw.json"
  target=test/"measurements"/source.name;shutil.copy2(source,target)
  payload=json.loads(target.read_text());assert len(payload["cells"])==10
  keys += [dict(task="depth",model=model,stage=cell["stage"],layer=cell["layer_index"],arm="raw") for cell in payload["cells"]]
  jobs.append(dict(kind="geometry",model=model,task="depth",arm="raw",out=str(target)))
 (test/"expected-inventory.json").write_text(json.dumps(dict(cells=keys,paired_comparison_cells=[])))
 (test/"completed-inventory.json").write_text(json.dumps(dict(completed=[],completed_paired=[])))
 os.chdir(test)
 try:
  processes=[multiprocessing.Process(target=register,args=(job,{})) for job in jobs]
  for process in processes:process.start()
  for process in processes:process.join();assert process.exitcode==0
  book=json.loads((test/"completed-inventory.json").read_text())
  assert book["completed_primary_cells"]==20 and len(book["completed"])==20 and not book["pending"]
  register(jobs[0],{})
  assert json.loads((test/"completed-inventory.json").read_text())["completed_primary_cells"]==20
  path=Path(jobs[0]["out"]);path.write_text(path.read_text()+"\n")
  try:register(jobs[0],{})
  except AssertionError:pass
  else:raise AssertionError("changed output hash was accepted")
 finally:os.chdir(original)
report=dict(passed=True,concurrent_atomic_updates_preserved=True,idempotent_verified_reuse=True,changed_output_hash_rejected=True,
 isolated_temporary_registry_only=True,real_measurement_inventory_untouched=True,driver_sha256=hashlib.sha256((root/"partition.py").read_bytes()).hexdigest())
(root/"registry-test-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
