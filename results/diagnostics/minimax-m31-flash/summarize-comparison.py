"""Generate paired per-Stage comparison tables from each interval's own measurements."""
import json,math
from pathlib import Path
import torch
torch.set_num_threads(2)
root=Path("/tmp/vtb-delivery/issues/21");base=root/"measurements";flash="minimax_m31_flash"
rows=[]
def read(name):return json.loads((base/"bootstrap"/name).read_text())
def add(task,stage,first,second,interval,filename,unit,scale=1,reverse=False):
 point=interval["point_estimate"];low=interval["lower"];high=interval["upper"]
 if reverse:point,low,high=-point,-high,-low
 assert all(math.isfinite(v) for v in (first,second,point,low,high))
 assert low<=high and abs((first-second)-point)<1e-5,(task,stage,first,second,point)
 rows.append({"task":task,"stage":stage,"flash":first*scale,"m3":second*scale,"delta_flash_minus_m3":point*scale,"lower":low*scale,"upper":high*scale,"unit":unit,"source":"bootstrap/"+filename,"points_from_same_paired_run":True})
for stage in ("tower","merged","projected"):
 name=f"semantic-m31-vs-m3-{stage}.json";d=read(name)
 add("Recognition, matched attention",stage,d["first"]["mean_accuracy"],d["second"]["mean_accuracy"],d["difference"],name,"percentage points",100)
 for arm in ("matched","raw"):
  for task in ("depth","normal"):
   name=f"geometry-{task}-diode-{arm}-m31-vs-m3-{stage}.json";d=read(name)
   add("DIODE "+("depth d1" if task=="depth" else "normals mean error")+", "+arm,stage,d["first"]["mean_metric"],d["second"]["mean_metric"],d["first_advantage"],name,"percentage points" if task=="depth" else "degrees",100 if task=="depth" else 1,task=="normal")
 for dataset in ("navi","scannet","spair"):
  name=f"correspondence-{dataset}-m31-vs-m3-{stage}.json";d=read(name);values=d["measurements"]
  if dataset=="spair":
   classes=sorted({r["class"] for r in values})
   first=sum(torch.tensor([r[flash] for r in values if r["class"]==c]).mean().item() for c in classes)/len(classes)
   second=sum(torch.tensor([r["minimax_m3"] for r in values if r["class"]==c]).mean().item() for c in classes)/len(classes)
   interval=d["class_macro_difference"]
  else:
   first=torch.tensor([r[flash] for r in values]).mean().item();second=torch.tensor([r["minimax_m3"] for r in values]).mean().item();interval=d["difference"]
  add(dataset.upper()+(" class-macro PCK" if dataset=="spair" else " recall"),stage,first,second,interval,name,"percentage points",100)
 name=f"perturbation-m31-vs-m3-{stage}.json";d=read(name)
 for condition,value in d["conditions"].items():
  values=value["measurements"];first=torch.tensor([r[flash] for r in values]).float().mean().item();second=torch.tensor([r["minimax_m3"] for r in values]).float().mean().item()
  add("Perturbation "+condition,stage,first,second,value["difference"],name,"percentage points",100)
for stage in ("tower","projected"):
 name=f"geometry-depth-kitti-matched-m31-vs-m3-{stage}.json";d=read(name)
 add("KITTI depth d1",stage,d["first"]["mean_metric"],d["second"]["mean_metric"],d["first_advantage"],name,"percentage points",100)
assert len(rows)==65,len(rows)
output={"difference":"Flash minus M3; negative is better for normal error, positive is better for accuracy and recall.","comparison":"Same architecture, weights pinned separately, identical isolated image inference at 448. Every point and interval comes from the same paired run.","rows":rows}
(root/"per-stage-deltas.json").write_text(json.dumps(output,indent=2)+"\n")
lines=["# MiniMax M3.1 Flash versus compatible M3 controls","",output["comparison"],"",output["difference"],"","## Main paired measurements","","| Task | Stage | Flash | M3 | Delta | 95% interval |","|---|---|---:|---:|---:|---|"]
for row in rows:
 if row["task"].startswith("Perturbation "):continue
 lines.append(f'| {row["task"]} | {row["stage"]} | {row["flash"]:.3f} | {row["m3"]:.3f} | {row["delta_flash_minus_m3"]:+.3f} | [{row["lower"]:+.3f}, {row["upper"]:+.3f}] |')
lines+=["","Accuracy, recall and d1 values are percentages; their deltas are percentage points. Normal errors and deltas are degrees. An interval spanning zero does not resolve the ordering. SPair uses class-macro paired intervals here; the standard within-model figure retains its historical per-pair protocol and reports a class-macro supplement.","","## All perturbation conditions","","Identity rows give the clean baselines on the same 300 test images. All conditions use readouts trained on 1,400 clean images, with the original primary reducer. Interpret condition differences relative to these clean readout baselines. The standardized full-recognition label-budget measurements are separate diagnostics.","","| Condition | Stage | Flash | M3 | Delta, pp | 95% interval, pp |","|---|---|---:|---:|---:|---|"]
for row in rows:
 if not row["task"].startswith("Perturbation "):continue
 lines.append(f'| {row["task"].removeprefix("Perturbation ")} | {row["stage"]} | {row["flash"]:.3f} | {row["m3"]:.3f} | {row["delta_flash_minus_m3"]:+.3f} | [{row["lower"]:+.3f}, {row["upper"]:+.3f}] |')
lines+=["","The complete JSON records each source payload. These paired estimates are not attached to points from a separate primary matrix. See image-isolation.md for the inference correction and source-reference limits.","","## Paper note","","MiniMax M3 and M3.1 Flash share the evaluated Tower architecture, preprocessing and Stage definitions. Their nominal image-size field does not affect these 448-pixel forwards. With identical image-independent inference and pinned weights, the paired per-Stage differences above isolate checkpoint training effects under the fixed probing protocols. The native packed attention path couples unrelated images, so this comparison uses fresh compatible M3 controls and preserves historical M3 payloads separately. Weight drift is reported for every Tower tensor and Projector tensor; concatenated layer drift is not monotonic and can be dominated by large-norm parameters."]
(root/"comparison-writeup.md").write_text("\n".join(lines)+"\n")
print("Wrote",len(rows),"paired per-Stage deltas",flush=True)

