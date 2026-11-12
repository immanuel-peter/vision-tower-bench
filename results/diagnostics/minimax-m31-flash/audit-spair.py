"""Audit every selected SPair source visibility count and classify every nonfinite metric."""
import hashlib,json,math,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path.cwd()/"scripts"))
from correspondence_run import SPairPairs
root=Path("/tmp/vtb-delivery/issues/21")
dataset=SPairPairs(Path("/tmp/vtb-delivery/data/correspondence/spair/SPair-71k"),448)
source={}
for i in range(len(dataset)):
 row=dataset[i]
 source[row["id"]]={"visible_keypoints":int(row["valid"].sum()),"class":row["class"],"viewpoint":row["viewpoint"]}
 if (i+1)%300==0:print(i+1,len(dataset),flush=True)
assert len(source)==len(dataset)==3600
excluded=[{"pair_id":key,"reason":"zero keypoints visible in both images after the common 448-pixel crop","visible_keypoints":0,"class":value["class"]} for key,value in source.items() if value["visible_keypoints"]==0]
report={"resolution":448,"selected_pairs":len(source),"source_visibility":source,"excluded":excluded,"nonempty_pairs":len(source)-len(excluded),"models":{}}
for model in ("minimax_m31_flash","minimax_m3"):
 path=root/"measurements/correspondence"/f"{model}_correspondence_spair.json"
 if not path.exists():continue
 payload=json.loads(path.read_text())
 assert {r["pair_id"] for r in payload["measurements"]}==set(source)
 for row in payload["measurements"]:
  count=source[row["pair_id"]]["visible_keypoints"]
  assert source[row["pair_id"]]["class"]==row["class"]
  for cell in row["cells"].values():
   assert cell["keypoints"]==count,(row["pair_id"],cell,count)
   assert math.isfinite(cell["pck_0.1"]) if count else math.isnan(cell["pck_0.1"])
 report["models"][model]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"verified_all_nonempty_finite":True,"verified_all_empty_source_counts":True,"excluded_ids":[r["pair_id"] for r in excluded]}
(root/"spair-visibility-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print("Selected",len(source),"nonempty",report["nonempty_pairs"],"explicitly verified zero-visible",len(excluded),flush=True)

