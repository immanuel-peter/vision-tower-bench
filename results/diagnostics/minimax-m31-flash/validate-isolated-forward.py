"""Independent single-image parent reference versus isolated adapter batches."""
import hashlib,inspect,json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as native
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
from vtb.adapters import minimax_m3 as mm
from vtb.shards import load_prefixed
torch.set_num_threads(8)
root=Path("/tmp/vtb-delivery/issues/21")
report={"reference":"Native HF classes loaded independently from pinned source tensors; each reference image is forwarded separately. This is not a second modeling implementation.","dtype":"bfloat16","device":"cuda:0","tolerance_max_abs":0,"native_source_sha256":hashlib.sha256(Path(inspect.getfile(native)).read_bytes()).hexdigest(),"adapter_source_sha256":hashlib.sha256(Path(inspect.getfile(mm)).read_bytes()).hexdigest(),"checkpoints":{}}
for cp,cls in ((mm.M3,mm.MiniMaxM3Adapter),(mm.FLASH,mm.MiniMaxM31FlashAdapter)):
 index=json.loads(Path(hf_hub_download(cp.repo,"model.safetensors.index.json",revision=cp.revision)).read_text())["weight_map"]
 raw={}
 for prefix in ("vision_tower.vision_model.","multi_modal_projector.","patch_merge_mlp."):
  shards=sorted({v for k,v in index.items() if k.startswith(prefix)})
  raw[prefix]=load_prefixed(cp.repo,shards,prefix,revision=cp.revision)
 tower_weights={k.replace("encoder.layers.","layers.").replace("embeddings.patch_embedding.","embeddings.proj."):v for k,v in raw["vision_tower.vision_model."].items()}
 projector_weights=dict(raw["multi_modal_projector."])
 projector_weights.update({k.replace("linear_","merge_linear_"):v for k,v in raw["patch_merge_mlp."].items()})
 config=MiniMaxM3VLConfig.from_pretrained(cp.repo,revision=cp.revision)
 ref=native.MiniMaxM3VLVisionModel._from_config(config.vision_config,dtype=torch.bfloat16)
 ref.load_state_dict(tower_weights,strict=True);ref=ref.to("cuda:0").eval()
 proj=native.MiniMaxM3VLMultiModalProjector(config).to(dtype=torch.bfloat16)
 proj.load_state_dict(projector_weights,strict=True);proj=proj.to("cuda:0").eval()
 adapter=cls(device="cuda:0",dtype=torch.bfloat16)
 rows=[]
 for resolution in (56,448):
  side=resolution//14;images=[];packed_images=[];expected=[];hashes=[]
  for image_index in range(8):
   pixels=np.random.default_rng(21+image_index).integers(0,256,(resolution,resolution,3),dtype=np.uint8)
   image=Image.fromarray(pixels);images.append(image);hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
   array=torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255
   array=(array-torch.tensor(mm.CLIP_MEAN)[:,None,None])/torch.tensor(mm.CLIP_STD)[:,None,None]
   patches=[]
   for y in range(0,side,2):
    for x in range(0,side,2):
     for dy in range(2):
      for dx in range(2):
       patch=array[:,(y+dy)*14:(y+dy+1)*14,(x+dx)*14:(x+dx+1)*14]
       patches.append(patch[:,None].expand(-1,2,-1,-1).reshape(-1))
   packed=torch.stack(patches)
   assert torch.equal(packed,mm.Preprocess(resolution)(image))
   packed_images.append(packed)
   captures={};handles=[]
   for layer in adapter.depth_points():
    handles.append(ref.layers[layer-1].register_forward_hook(lambda module,inp,out,layer=layer:captures.__setitem__(layer,out.detach())))
   with torch.inference_mode():
    output=ref(packed.to("cuda:0",torch.bfloat16),torch.tensor([[1,side,side]],device="cuda:0")).last_hidden_state
    projected=proj(output.reshape(-1,1280)).cpu()
   for h in handles:h.remove()
   sample={}
   for layer,values in captures.items():
    values=values.reshape(-1,1280).cpu();raster=torch.empty(side,side,1280,dtype=torch.bfloat16);at=0
    for y in range(0,side,2):
     for x in range(0,side,2):
      for dy in range(2):
       for dx in range(2):
        raster[y+dy,x+dx]=values[at];at+=1
    sample[("tower",layer)]=raster.reshape(-1,1280)
    if layer==32:
     sample[("merged",32)]=torch.stack([raster[y:y+2,x:x+2].reshape(5120) for y in range(0,side,2) for x in range(0,side,2)])
   sample[("projected",32)]=projected;expected.append(sample)
  for order in ([0],[0,1],list(range(8)),[1,0],[0,7]):
   adapter.resolution=resolution
   actual=list(adapter.extract(mm.collate([packed_images[i] for i in order]),[str(i) for i in order]))
   comparisons=[]
   for batch in actual:
    reference=torch.stack([expected[i][(batch.stage,batch.layer_index)] for i in order])
    delta=(batch.tokens.float()-reference.float()).abs().max().item()
    assert delta==0,(cp.name,resolution,order,batch.stage,batch.layer_index,delta)
    comparisons.append({"stage":batch.stage,"layer":batch.layer_index,"shape":list(batch.tokens.shape),"max_abs":delta,"bit_exact":True})
   rows.append({"resolution":resolution,"image_order":order,"image_sha256":[hashes[i] for i in order],"comparisons":comparisons})
   print(cp.name,resolution,order,"all slices exact",flush=True)
 report["checkpoints"][cp.name]={"repo":cp.repo,"revision":cp.revision,"cases":rows}
 del adapter,ref,proj,raw,tower_weights,projector_weights
 torch.cuda.empty_cache()
(root/"isolated-parent-forward.json").write_text(json.dumps(report,indent=2)+"\n")

