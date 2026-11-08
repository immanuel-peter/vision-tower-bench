from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess,tempfile
root=Path(__file__).parent
with tempfile.TemporaryDirectory(dir=root,prefix="admission-test-") as folder:
 test=Path(folder);shutil.copy2(root/"admission_hook.py",test/"sitecustomize.py")
 policy=test/"policy.json";proof=test/"proof.json";out=test/"never-created.json"
 policy.write_text(json.dumps(dict(enabled=True,visible_device="",targets=[str(out)],proof=str(proof))))
 env=os.environ|{"CUDA_VISIBLE_DEVICES":"","PYTHONPATH":str(test)+":"+str(Path.cwd()),"VTB_ADMISSION_POLICY":str(policy),"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"}
 args=[str(Path.cwd()/".venv/bin/python"),"-m","vtb.geometry_run","--out",str(out)]
 matching=subprocess.run(["bash","-ic",shlex.join(args)],env=env,capture_output=True,text=True)
 assert matching.returncode==75 and proof.exists() and not out.exists(),matching
 actual=json.loads(proof.read_text());assert actual["torch_not_imported"] and actual["cuda_model_never_started"]
 args=[str(Path.cwd()/".venv/bin/python"),"-c","print(12345)"]
 unrelated=subprocess.run(["bash","-ic",shlex.join(args)],env=env,capture_output=True,text=True)
 assert unrelated.returncode==0 and unrelated.stdout.strip()=="12345",unrelated
 report=dict(passed=True,matching_unstarted_process_exit=75,unrelated_process_exit=0,no_model_output_created=True,torch_not_imported=True,cuda_not_initialized=True,
  hook_sha256=hashlib.sha256((root/"admission_hook.py").read_bytes()).hexdigest(),proof=actual)
 (root/"admission-test-report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({key:value for key,value in report.items() if key!="proof"}))
