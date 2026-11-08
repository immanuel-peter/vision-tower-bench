"""Independently check the completed mixed-device raw matrices and archive evidence."""
import hashlib,json,shutil,datetime
from pathlib import Path
from scripts.validate_kimi_results import validate_probe
root=Path("/tmp/vtb-delivery/issues/24")
public=Path("results/diagnostics/kimi-k31-pro-checkpoint")
archive=public/"raw-attention-recognition";archive.mkdir(exist_ok=True)
progress=json.loads((root/"gpu-raw-recognition-progress.json").read_text())
assert progress["complete"] and len(progress["results"])==20
ids=json.loads((root/"ids-imagenet100.json").read_text())["ids"]
splits=json.loads((root/"splits-imagenet100.json").read_text())
labels=json.loads(Path("/tmp/vtb-delivery/data/imagenet100/validation_labels.json").read_text())["labels"]
models={"kimi_k31_pro":"immanuelpeter/MoonViT-K3.1-Pro","moonvit_v2":"immanuelpeter/MoonViT-V2"}
matrices={};reports=[]
for model,model_id in models.items():
 path=root/"measurements/recognition"/f"{model}_probe_attention_raw.json"
 matrix=json.loads(path.read_text());matrices[model]=matrix
 audit=validate_probe(matrix,model_id,ids,"recognition","raw","attention",expected_splits=splits)
 for cell in matrix["cells"]:
  source=Path(cell["source_fit"]["file"])
  assert hashlib.sha256(source.read_bytes()).hexdigest()==cell["source_fit"]["sha256"]
  single=json.loads(source.read_text());original=single["cells"][0]
  assert cell["source_fit"]["protocol"]==single["protocol"]
  assert {key:value for key,value in cell.items() if key!="source_fit"}==original
  assert cell["test_labels"]==[labels[image] for image in cell["test_image_ids"]]
  assert cell["fit_threads"]==8
  assert cell["fit_device"]==("cpu" if cell["stage"]=="tower" and cell["layer_index"] in (3,7) else "cuda:0")
 audit.update(model=model,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              cpu_cells=2,gpu0_cells=8,threads_per_cell=8)
 reports.append(audit)
for a in matrices["kimi_k31_pro"]["cells"]:
 b=next(row for row in matrices["moonvit_v2"]["cells"] if (row["stage"],row["layer_index"])==(a["stage"],a["layer_index"]))
 assert a["fit_device"]==b["fit_device"] and a["fit_threads"]==b["fit_threads"]
report=dict(passed=True,cells=20,complete_recognition_suite_cells=80,cpu_fits_reused=4,gpu0_new_fits=16,
 new_cell_run_and_validation_seconds=sum(row["seconds"] for row in progress["results"] if not row["resumed"]),
 same_device_and_threads_within_every_sibling_pair=True,per_cell_original_protocols_preserved=True,
 matrices=reports,verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/"raw-recognition-complete-audit.json").write_text(json.dumps(report,indent=2)+"\n")
for name in ("gpu-raw-recognition-progress.json","raw-recognition-complete-audit.json","gpu_semantic_raw_resume.py","archive_raw_recognition.py"):
 shutil.copy2(root/name,archive/name)
rows=[json.loads(line) for line in (root/"execution.jsonl").read_text().splitlines()]
(archive/"suite-start.json").write_text(json.dumps(next(row for row in reversed(rows) if row.get("phase")=="gpu0-remaining-suite" and row["event"]=="start"),indent=2)+"\n")
handoff=json.loads((root/"handoff.json").read_text())
(public/"ownership-boundary.json").write_text(json.dumps(handoff["ownership_boundary"],indent=2)+"\n")
print(json.dumps({key:value for key,value in report.items() if key!="matrices"}))
