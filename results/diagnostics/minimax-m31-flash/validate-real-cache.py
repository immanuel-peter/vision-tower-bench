"""Recompute first/middle/last real inputs and compare every cached slice exactly."""
import hashlib,json
from pathlib import Path
from PIL import Image
from safetensors import safe_open
import torch
from vtb import cache
from vtb.adapters import minimax_m3 as mm
torch.set_num_threads(8)
root=Path("/tmp/vtb-delivery/issues/21");data=Path("/tmp/vtb-delivery/data");report={}
for cp,cls in ((mm.FLASH,mm.MiniMaxM31FlashAdapter),(mm.M3,mm.MiniMaxM3Adapter)):
 adapter=cls(device="cuda:0")
 for dataset in ("recognition","diode","kitti"):
  folder=data/"imagenet100/validation" if dataset=="recognition" else data/dataset/"val"
  files=sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg",".png"))
  chosen=[files[i] for i in (0,len(files)//2,len(files)-1)]
  inputs=[]
  for path in chosen:
   with Image.open(path) as image:inputs.append(adapter.preprocess()(image.convert("RGB")))
  ids=[p.stem for p in chosen];tag="pool4" if dataset=="recognition" else "full"
  run=root/"features"/dataset/f"{cp.name}_448_{tag}";slices=set(cache.slices(run));rows=[]
  for batch in adapter.extract(mm.collate(inputs),ids):
   if (batch.stage,batch.layer_index) not in slices:continue
   if dataset=="recognition":batch=batch.pooled(4)
   found={}
   for shard in cache.shards(run,batch.stage,batch.layer_index):
    with safe_open(shard,framework="pt") as f:
     names=f.metadata()["image_ids"].split("\n")
     for image_id in ids:
      if image_id in names:found[image_id]=f.get_slice("tokens")[names.index(image_id)]
   assert set(found)==set(ids)
   for i,image_id in enumerate(ids):
    assert torch.equal(batch.tokens[i],found[image_id]),(cp.name,dataset,image_id,batch.stage,batch.layer_index)
   rows.append({"stage":batch.stage,"layer":batch.layer_index,"images":len(ids),"bit_exact":True})
  assert len(rows)==len(slices)
  report[f"{cp.name}/{dataset}"]={"image_ids":ids,"input_sha256":{p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in chosen},"comparisons":rows}
  print(cp.name,dataset,"real-input cache recomputation exact",flush=True)
 del adapter
 torch.cuda.empty_cache()
(root/"real-input-cache-parity.json").write_text(json.dumps(report,indent=2)+"\n")

