"""Publish only a fully content-validated measurement inventory into additive repo paths."""
import hashlib,json,shutil
from pathlib import Path
root=Path("/tmp/vtb-delivery/issues/21");source=root/"measurements";repo=Path.cwd();results=repo/"results"
diag=results/"diagnostics/minimax-m31-flash";control=diag/"m3-image-isolation"
audits=[json.loads((root/name).read_text()) for name in ("measurement-validation.json","measurement-validation-minimax_m3.json")]
assert all(a["complete"] and not a["missing"] for a in audits)
for model in ("minimax_m31_flash","minimax_m3"):
 value=json.loads((root/f"perturbation-cache-audit-{model}.json").read_text())
 assert value["complete"] and value["shards"]==156 and len(value["conditions"])==13 and len(value["common_image_ids"])==2000
paired=json.loads((root/"paired-source-validation.json").read_text());assert len(paired)==20
def destination(path):
 rel=path.relative_to(source);name=path.name;kind=rel.parts[0]
 if kind=="standardized":
  if name=="minimax_m31_flash_label_budget_standardized.json":return diag/"label_budget_standardized.json"
  if name=="minimax_m3_label_budget_standardized.json":return control/"label_budget_standardized.json"
  return (control if name.startswith("minimax_m3_") else diag)/rel
 is_control=name=="minimax_m3.json" or name.startswith("minimax_m3_") or name.endswith("-minimax_m3-projected-vs-tower.json") or name.endswith("-minimax_m3-projected-minus-tower.json") or name=="perturbation-minimax_m3.json"
 return (control if is_control else results)/rel
provenance={"image_inference":"independent native single-image Tower and Projector forwards inside each external batch","source_reference_limit":"Native HF classes with independent pinned-source loading, not a second modeling implementation or a full LM forward.","inputs":{"prepared_manifest":"/tmp/vtb-delivery/data-sha256.txt","prepared_manifest_sha256":hashlib.sha256(Path("/tmp/vtb-delivery/data-sha256.txt").read_bytes()).hexdigest()},"checkpoints":{"minimax_m3":{"repo":"MiniMaxAI/MiniMax-M3","revision":"f0e1c1e04d40177e4673a22097036854f536e9c0"},"minimax_m31_flash":{"repo":"MiniMaxAI/MiniMax-M3.1-Flash","revision":"54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"}},"hf_release":json.loads((root/"hf-publication.json").read_text()),"files":{}}
planned=[(path,destination(path)) for path in sorted(source.rglob("*.json"))]
assert len(planned)==103,len(planned)
assert len({target for _,target in planned})==len(planned)
identity_report=json.loads((root/"identity-evidence-materialization.json").read_text())
originals=[]
for model in ("minimax_m31_flash","minimax_m3"):
 raw=root/"raw-probe-payloads"/f"{model}_perturbation.json"
 assert hashlib.sha256(raw.read_bytes()).hexdigest()==identity_report[model]["original_sha256"]
 augmented=json.loads((source/"perturbation"/raw.name).read_text())
 header=augmented.pop("identity_evidence")
 assert hashlib.sha256((source/"perturbation"/raw.name).read_bytes()).hexdigest()==identity_report[model]["augmented_sha256"]
 assert header["materialization_script_sha256"]==hashlib.sha256((root/"postprocess.py").read_bytes()).hexdigest()
 cache_audit=json.loads((root/f"perturbation-cache-audit-{model}.json").read_text())
 assert header["source_runner_sha256"]==cache_audit["execution"]["source_hashes"]["scripts/perturbation_run.py"]
 assert header["original_sha256"]==identity_report[model]["original_sha256"]
 for cell in augmented["cells"]:
  identity=cell["measurements"].pop("identity")
  assert all(row["clean"]==row["perturbed"] for row in identity)
 assert augmented==json.loads(raw.read_text()),"Original protocol, score or transformed measurement changed"
 target=diag/"raw-probe-payloads"/raw.name
 assert not target.exists(),"Refuse to replace original runtime evidence: "+str(target)
 originals.append((raw,target))
for path,target in planned:
 assert not target.exists(),"Refuse to replace any existing published measurement: "+str(target)
for path,target in planned:
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(path,target)
 provenance["files"][str(target.relative_to(repo))]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"raw_source":str(path),"kind":path.relative_to(source).parts[0]}
assert len(provenance["files"])==103,len(provenance["files"])
provenance["original_runtime_payloads"]={}
for path,target in originals:
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
 provenance["original_runtime_payloads"][str(target.relative_to(repo))]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"purpose":"Exact runner output before lossless identity-row materialization; scores/protocol unchanged."}
(diag/"measurement-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
for name in ("measurement-validation.json","measurement-validation-minimax_m3.json","paired-source-validation.json","hf-publication.json","hf-current-release-reload.json","spair-visibility-audit.json","execution.jsonl","expected-inventory.json","real-input-cache-parity.json","selection-manifest.json","semantic-completion-audit.json","transfer-throughput-completion-audit.json","publication-plan-audit.json","identity-evidence-materialization.json","perturbation-cache-audit-minimax_m31_flash.json","perturbation-cache-audit-minimax_m3.json"):
 shutil.copyfile(root/name,diag/name)
# Compact duplicate split-ID lists, while retaining the complete runtime audit files.
cache_manifests={}
for path in sorted(root.glob("cache-audit-*.json")):
 value=json.loads(path.read_text());common=None
 for entry in value["slices"].values():
  splits=entry.pop("split_ids")
  if common is None:common=splits
  assert splits==common
  entry["split_ids_reference"]="common_split_ids"
 value["common_split_ids"]=common;cache_manifests[path.stem]=value
assert len(cache_manifests)==6
(diag/"cache-manifests.json").write_text(json.dumps(cache_manifests,indent=2)+"\n")
# Add measurement-cell manifests without changing any historical entry.
index_path=results/"index.json";index=json.loads(index_path.read_text())
formats={
 "recognition":(f"{{model}}_probe_{{x}}_{{y}}.json",[(x,y) for x in ("attention","mean") for y in ("raw","matched")],10),
 "geometry":(f"{{model}}_geometry_{{x}}_{{y}}.json",[(x,y) for x in ("depth","normal") for y in ("raw","matched")],10),
 "label-budget":(f"{{model}}_probe_{{x}}_matched_{{y}}.json",[(x,y) for x in ("attention","mean") for y in ("f001","f005","f020","f10")],1),
 "correspondence":("{model}_correspondence_{x}.json",[(x,"") for x in ("navi","scannet","spair")],3),
 "perturbation":("{model}_perturbation.json",[("","")],3),
 "kitti":("{model}_geometry_depth_matched.json",[("","")],2),
 "standardized":("{model}_probe_attention_matched_{x}.json",[(x,"") for x in ("f001","f005","f02","f10")],1),
}
for model in ("minimax_m31_flash","minimax_m3"):
 for kind,(pattern,values,cells) in formats.items():
  directory=kind if model=="minimax_m31_flash" else f"diagnostics/minimax-m31-flash/m3-image-isolation/{kind}"
  if kind=="standardized":directory=("diagnostics/minimax-m31-flash/standardized" if model=="minimax_m31_flash" else "diagnostics/minimax-m31-flash/m3-image-isolation/standardized")
  name=f"manifests/expected_cells_{model}_{kind.replace('-','_')}.tsv"
  entries=[(pattern.format(model=model,x=x,y=y),cells) for x,y in values]
  (results/name).write_text("".join(f"{filename}\t{count}\n" for filename,count in entries))
  index["cell_manifests"].append({"path":name,"results_directory":directory+"/","cohort":"minimax_m31_flash" if model=="minimax_m31_flash" else "minimax_m3_image_isolation"})
index["cohorts"]["minimax_m31_flash"]["description"]="Full image-independent 448-pixel suite for MiniMax M3.1 Flash, with fresh pinned M3 controls under the same inference and readout protocols."
index["experiments"]["minimax_m3_image_isolation"]["status"]="Complete primary matrices, standardized label budgets, correspondence, perturbations, transfer, throughput and paired within-model comparisons. Historical files preserved."
index["experiments"]["minimax_m31_flash_validation"]["status"]="Complete source/release identity, independent image-wise HF parity, input/cache validation, full primary inventory and paired comparisons."
index["experiments"]["minimax_m31_flash_suite"]={"path":"diagnostics/minimax-m31-flash/","protocol":"All primary protocols at 448, image-independent inference, full eight-depth sweeps, opt-in standardized label-budget diagnostics, matched pinned M3 controls and same-run paired intervals.","status":"Complete; see measurement-provenance.json and both content-validation reports.","cohort":"minimax_m31_flash"}
index_path.write_text(json.dumps(index,indent=2)+"\n")
print("Published",len(provenance["files"]),"validated JSON payloads and additive source manifests",flush=True)

