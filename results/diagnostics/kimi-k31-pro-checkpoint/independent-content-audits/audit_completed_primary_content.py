"""Revalidate actual completed payload contents and inventory-to-source mappings."""
import datetime,hashlib,json,math
from collections import Counter
from pathlib import Path
from scripts.validate_kimi_results import validate_probe,validate_correspondence,validate_perturbation,finite_numbers

root=Path("/tmp/vtb-delivery/issues/24")
book=json.loads((root/"completed-inventory.json").read_text())
expected=json.loads((root/"expected-inventory.json").read_text())
models={"kimi_k31_pro":"immanuelpeter/MoonViT-K3.1-Pro","moonvit_v2":"immanuelpeter/MoonViT-V2"}
releases={"kimi_k31_pro":"6ab95e2daf86e8100be2e712a7a71c0fba26dac4","moonvit_v2":"bcaedfccd1f07d23ab20de8a0a168668c27cacb3"}
def read(path):return json.loads(Path(path).read_text())
ids={name:read(root/f"ids-{name}.json")["ids"] for name in ("imagenet100","diode","kitti","perturbation")}
splits={name:read(root/f"splits-{name}.json") for name in ids}
visibility=read(root/"spair-visibility-audit.json")
conditions=sorted({cell["condition"] for cell in expected["cells"] if cell["task"]=="perturbation"})
transform_hash=hashlib.sha256(Path("/tmp/vtb-delivery/data/perturbation/transforms.manifest.json").read_bytes()).hexdigest()
source_cache={};audited={};records=[]
for entry in book["completed"]:
 key=entry["cell"];source=Path(entry["source"]);task=key["task"];model=key["model"]
 assert key in expected["cells"]
 if str(source) not in source_cache:
  source_cache[str(source)]=(read(source),hashlib.sha256(source.read_bytes()).hexdigest())
 payload,digest=source_cache[str(source)]
 assert digest==entry["sha256"]
 signature=(str(source),task,key.get("arm"),key.get("readout"),key.get("fraction"))
 if signature not in audited:
  if task in {"recognition","label_budget","standardized_label_budget","depth","normal","kitti"}:
   semantic=task in {"recognition","label_budget","standardized_label_budget"}
   dataset="imagenet100" if semantic else ("kitti" if task=="kitti" else "diode")
   kind={"label_budget":"label-budget","standardized_label_budget":"standardized"}.get(task,"geometry" if task in {"depth","normal"} else task)
   split=splits[dataset]
   if task in {"label_budget","standardized_label_budget"}:
    fraction=key["fraction"]
    split=read(root/f"splits-imagenet100-f{fraction}.json")
   scoped={(key["stage"],key["layer"])} if task=="recognition" and len(payload["cells"])==1 else None
   audit=validate_probe(payload,models[model],ids[dataset],kind,key["arm"],key.get("readout","attention"),key.get("fraction",1.0),expected_splits=split,expected_slices=scoped)
   if task in {"depth","normal"}:assert payload["task"]==task
   assert payload["feature_provenance"]["details"]["release_revision"]==releases[model]
  elif task=="perturbation":
   audit=validate_perturbation(payload,models[model],ids["perturbation"],conditions,splits["perturbation"],releases[model],transform_hash,device="cpu",threads=4)
  elif task in {"navi","scannet","spair"}:
   selected=read(root/f"pairs-{task}.json")["ids"]
   audit=validate_correspondence(payload,selected,visibility if task=="spair" else None)
   assert payload["model_id"]==models[model]
   for cell in payload["cells"]:
    stage=cell["stage"]+":27"
    for metric,value in cell.items():
     if metric in {"stage","layer_index","by_class","macro_pck_0.1"}:continue
     values=[row["cells"][stage][metric] for row in payload["measurements"] if math.isfinite(row["cells"][stage][metric])]
     assert values and abs(value-math.fsum(values)/len(values))<1e-6
    if task=="spair":
     classes={row["class"] for row in payload["measurements"]}
     assert set(cell["by_class"])==classes
     for category in classes:
      values=[row["cells"][stage]["pck_0.1"] for row in payload["measurements"] if row["class"]==category and row["cells"][stage]["keypoints"]>0]
      assert values and abs(cell["by_class"][category]-math.fsum(values)/len(values))<1e-6
     assert abs(cell["macro_pck_0.1"]-math.fsum(cell["by_class"].values())/len(classes))<1e-6
  elif task=="throughput":
   assert finite_numbers(payload) and payload["model_id"]==models[model] and payload["images"]==768
   assert payload["resolution"]==448 and payload["batch_size"]==1 and payload["workers"]==2
   assert payload["configured_patch_size"]==14 and payload["tower_tokens_per_image"]==1024
   runtime=read(source.with_suffix(".runtime.json"))["adapters"][0]
   assert runtime["exact_immutable_identity"] and runtime["identity_tensors"]==168 and runtime["checkpoint"]["bundle_revision"]==releases[model]
   selected=read(root/"throughput-source-selection.json");assert len(selected["ids"])==len(set(selected["ids"]))==768
   audit=dict(passed=True,images=768,batch_size=1,actual_source_selection_sha256=hashlib.sha256((root/"throughput-source-selection.json").read_bytes()).hexdigest())
  else:raise AssertionError(task)
  audited[signature]=dict(source=str(source),task=task,sha256=digest,validation=audit)
 if task!="throughput":
  cells=[cell for cell in payload["cells"] if cell["stage"]==key["stage"] and cell["layer_index"]==key.get("layer",27)]
  assert len(cells)==1
  if task=="perturbation":assert len([row for row in cells[0]["conditions"] if row["condition"]==key["condition"]])==1
 records.append(dict(cell=key,source=str(source),sha256=digest))
keys=[json.dumps(row["cell"],sort_keys=True) for row in records]
assert len(keys)==len(set(keys))==book["completed_primary_cells"]
assert set(keys)<={json.dumps(cell,sort_keys=True) for cell in expected["cells"]}
report=dict(passed=True,completed_primary_cells=len(records),required_primary_cells=286,complete_primary_inventory=len(records)==286,
 actual_cell_to_source_mapping_verified=True,source_payload_groups=len(audited),counts_by_task=dict(Counter(row["cell"]["task"] for row in records)),
 actual_content_validations=list(audited.values()),actual_source_records=records,
 verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/"completed-primary-independent-content-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({key:value for key,value in report.items() if key not in {"actual_content_validations","actual_source_records"}}))
