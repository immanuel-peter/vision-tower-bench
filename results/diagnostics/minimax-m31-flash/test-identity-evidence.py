"""Check lossless identity recovery, unchanged metrics, idempotence and mismatch rejection."""
import copy,hashlib,json,tempfile,textwrap
from pathlib import Path
import torch
torch.set_num_threads(2)
base=Path("/tmp/vtb-delivery/issues/21");models=("minimax_m31_flash","minimax_m3")
source=(base/"postprocess.py").read_text()
section=source.split("if phase.section in (\"all\",\"remaining\"):\n",1)[1]
block=textwrap.dedent(section.split(" first=results/\"perturbation\"/f\"{flash}_perturbation.json\";second=",1)[0])
def setup(root,bad=False):
 results=root/"measurements";(results/"perturbation").mkdir(parents=True)
 originals={}
 for model in models:
  cells=[]
  for stage in ("tower","merged","projected"):
   rows=[{"image_id":str(i),"clean":[float((i+s)%2) for s in range(3)],"perturbed":[0.,0.,0.]} for i in range(300)]
   conditions=["identity"]+[f"condition{i}" for i in range(12)]
   cell={"stage":stage,"conditions":[{"condition":name,"accuracy":.5 if name=="identity" else 0.,"accuracy_by_seed":[.5]*3 if name=="identity" else [0.]*3} for name in conditions],"measurements":{name:copy.deepcopy(rows) for name in conditions[1:]}}
   cells.append(cell)
  if bad:cells[0]["measurements"]["condition1"][0]["clean"][0]=1.
  payload={"protocol":{"seeds":3,"training":"unchanged"},"cells":cells};originals[model]=copy.deepcopy(payload)
  (results/"perturbation"/f"{model}_perturbation.json").write_text(json.dumps(payload))
 return results,originals
def run(root,results):
 exec(block,{"root":root,"results":results,"models":models,"json":json,"hashlib":hashlib,"torch":torch,"Path":Path,"__file__":str(base/"postprocess.py")})
with tempfile.TemporaryDirectory(dir=base,prefix="identity-evidence-test-") as directory:
 root=Path(directory);results,originals=setup(root)
 run(root,results);first={}
 for model in models:
  path=results/"perturbation"/f"{model}_perturbation.json";first[model]=path.read_bytes();d=json.loads(first[model]);d.pop("identity_evidence")
  for cell in d["cells"]:
   identity=cell["measurements"].pop("identity");assert all(row["clean"]==row["perturbed"] for row in identity)
   assert len(identity)==300
  assert d==originals[model],"Original scores/protocol/transformed rows changed"
  assert json.loads((root/"raw-probe-payloads"/path.name).read_text())==originals[model]
 run(root,results)
 assert all((results/"perturbation"/f"{model}_perturbation.json").read_bytes()==first[model] for model in models)
 negative=root/"negative";negative.mkdir();negative_results,_=setup(negative,bad=True)
 try:run(negative,negative_results)
 except AssertionError:pass
 else:raise AssertionError("Inconsistent clean vectors accepted")
print("PASS:6Stage identity reconstructions, original metrics/protocol preservation, exact raw backups, idempotence, inconsistent-vector rejection")
