import importlib.util,json,hashlib,datetime,collections,math
from pathlib import Path
import torch
from safetensors import safe_open
E=Path("/tmp/vtb-delivery/issues/22");ROOT=Path("/home/shadeform/.t3/worktrees/vision-tower-bench/t3-b75c6e2e");R=ROOT/"results";M="qwen38_flash_next"
assert torch.get_num_threads()==6
spec=importlib.util.spec_from_file_location("suite",E/"suite.py");s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);s.build();assert len(s.jobs)==153
history=[json.loads(line) for line in (E/"execution.jsonl").read_text().splitlines()];records={}
for row in history:
 if row.get("exit_code")==0:records[row.get("job")]=row
checks=[];cache_shards={};failures={}
for job in s.jobs:
 try:
  hashes=s.validate(job);record=records[job["name"]];assert record["output_hashes"]==hashes
  if job["kind"] in ["cache","perturbation-cache"]:
   for path,digest in hashes.items():
    with safe_open(path,framework="pt") as f:assert f.get_tensor("tokens").dtype==torch.bfloat16
    cache_shards[path]=digest
  checks.append({"job":job["name"],"kind":job["kind"],"files":len(hashes),"output_hashes":hashes})
 except Exception as error:failures[job["name"]]=repr(error)
 print("AUDIT",job["name"],"FAIL" if job["name"] in failures else "PASS",flush=True)
assert not failures,failures
splits=json.loads((R/"evidence/qwen38-flash-next/splits.json").read_text())["datasets"]
primary={};standard_sources={}
def load(path):
 path=R/path;standard_sources[str(path.relative_to(ROOT))]=s.sha(path);return json.loads(path.read_text())
for task,readouts in [("recognition",["attention","mean"]),("geometry",["depth","normal"])]:
 cells=[]
 for readout in readouts:
  for arm in ["raw","matched"]:
   suffix="probe" if task=="recognition" else "geometry";p=load(f"{task}/{M}_{suffix}_{readout}_{arm}.json");assert len(p["cells"])==10
   cells.extend((readout,arm,c["stage"],c["layer_index"]) for c in p["cells"])
 assert len(cells)==len(set(cells))==40;primary[task+"_cells"]=40
p=load(f"kitti/{M}_geometry_depth_matched.json");assert len(p["cells"])==2;primary["kitti_cells"]=2
label_cells=[]
for readout in ["attention","mean"]:
 for tag,fraction,count in [("f001",.01,100),("f005",.05,455),("f020",.2,1820),("f10",1.,9100)]:
  p=load(f"label-budget/{M}_probe_{readout}_matched_{tag}.json");assert p["label_fraction"]==fraction and p["train_images"]==count and len(p["cells"])==1;label_cells.append((readout,tag))
assert len(set(label_cells))==8;primary["label_budget_cells"]=8
correspondence={};source_pair_audit=json.loads((E/"pair-selection.json").read_text())["datasets"]
for model in [M,"qwen3_5"]:
 total=0
 for dataset,count in [("navi",555),("scannet",1500),("spair",3600)]:
  path=f"correspondence/{model}_correspondence_{dataset}.json" if model==M else f"evidence/qwen38-flash-next/controls/{model}_correspondence_{dataset}.json"
  p=load(path);assert p["pairs"]==len(p["measurements"])==count and len(p["cells"])==3;total+=count
  expected_zero=set(source_pair_audit[dataset].get("zero_visible_keypoint_ids",[]));nonempty=0
  for row in p["measurements"]:
   for metric in row["cells"].values():
    if dataset=="spair" and row["pair_id"] in expected_zero:assert metric["keypoints"]==0 and math.isnan(metric["pck_0.1"])
    else:s.finite(metric)
   nonempty+=row["pair_id"] not in expected_zero
  correspondence[model+"/"+dataset]={"selected_pairs":count,"stage_cells":3,"finite_pair_intersection":nonempty,"excluded_ids":sorted(expected_zero)}
 assert total==5655
primary["correspondence_stage_cells_new_model"]=9;primary["correspondence_selected_pair_records_both_models"]=11310
for model in [M,"qwen3_5"]:
 path=f"perturbation/{model}_perturbation.json" if model==M else f"evidence/qwen38-flash-next/controls/{model}_perturbation.json"
 p=load(path);assert len(p["cells"])==3 and all(len(c["conditions"])==13 for c in p["cells"])
primary["perturbation_condition_cells_new_model"]=39;primary["perturbation_condition_cells_fresh_control"]=39
p=load(f"label-budget/{M}_throughput.json");assert p["images"]==768;primary["throughput_images"]=768
bootstrap_jobs=[j for j in s.jobs if "/measurements/bootstrap/" in j["output"]];assert len(bootstrap_jobs)==33
coverage=collections.Counter()
for job in bootstrap_jobs:
 name=job["name"]
 group="correspondence" if any(d in name for d in ["navi","scannet","spair"]) else "perturbation" if "perturbation" in name else "recognition" if "recognition" in name else "kitti" if "kitti" in name else "geometry"
 coverage[group]+=1
assert dict(coverage)=={"geometry":10,"correspondence":12,"recognition":4,"kitti":3,"perturbation":4},coverage
for f in ["parity-cpu.json","parity-cpu-bf16.json","parity-gpu-bf16.json","release-reload.json"]:
 p=json.loads((E/f).read_text());assert p["source_revision"]=="de4b8e4d43b917e7706784d8bb445c9af86a3540" and p["reference_class"]=="Qwen4ExpVisionModel";assert len(p["identity"])==333 and len(p["cases"])==20 and all(c["bit_exact"] and c["max_abs"]==0 for c in p["cases"]);assert p["identity"]["merger.linear_fc2.weight"]["shape"]==[2560,4608]
r=json.loads((E/"release-reload.json").read_text());assert r["release_revision"]=="120ed5d5baf11cb478099350e94218ee3cbed3df" and r["native_state_tensor_count"]==333;assert all(v["shape"]==[18] and v["dtype"]=="torch.float32" for v in r["native_nonpersistent_buffers"].values())
assert s.sha(ROOT/"hf/Qwen3.8-Flash-Next-Vision/LICENSE")==r["release_files"]["LICENSE"]
drift=json.loads((E/"drift.json").read_text())["tensors"];assert len(drift)==333;shared=[d for d in drift.values() if "cosine" in d];assert len(shared)==331 and all(d["equal"] is False for d in shared)
diag={"standardized_label_budget_cells":4,"standardized_perturbation_condition_cells_both_models":78,"raw_seed0_replays":6,"full_channel_stage_records":6,"paired_standardized_perturbation_records":4}
for tag in ["f001","f005","f020","f10"]:load(f"diagnostics/standardized-label-budget/{M}_probe_attention_matched_{tag}.json")
for model in [M,"qwen3_5"]:
 p=load(f"diagnostics/qwen38-flash-next-conditioning/conditioning-{model}-sensitivity.json");assert len(p["cells"])==3 and len(p["training_losses"])==42
 p=load(f"diagnostics/qwen38-flash-next-conditioning/conditioning-{model}-audit.json");assert len(p["statistics"])==3
index=json.loads((R/"index.json").read_text());assert set(index["cohorts"]["v3"]["models"])=={"nemotron35_super","nemotron35_bolt","dinov3","mimo26_pro","mimo26_flash"};assert index["cohorts"]["checkpoint_additions"]["models"]==[M]
report={"completed_at":datetime.datetime.now(datetime.UTC).isoformat(),"all_jobs_content_validated":153,"cache_jobs":8,"measurement_jobs":145,"failures":failures,"primary":primary,"paired_bootstrap_records":dict(coverage),"paired_bootstrap_total":33,"diagnostics_separate_from_primary":diag,"correspondence":correspondence,"cache_tensor_shards":len(cache_shards),"cache_tensor_bytes":sum(Path(p).stat().st_size for p in cache_shards),"cache_shards_sha256":cache_shards,"standard_source_hashes":standard_sources,"hf_immutable_revision":r["release_revision"],"source_tensor_count":333,"source_and_release_forward_cases_per_dtype":20,"historical_cohorts_preserved":True,"job_checks":checks}
(E/"requirement-content-audit.json").write_text(json.dumps(report,indent=2)+"\n");print("FULLREQUIREMENTAUDITPASS",json.dumps({k:v for k,v in report.items() if k not in ["job_checks","cache_shards_sha256","standard_source_hashes","correspondence"]}))
