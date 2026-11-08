import hashlib,json,time
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch

torch.set_num_threads(2)
E=Path('/tmp/vtb-delivery/issues/21')
W=Path('/home/shadeform/.t3/worktrees/vision-tower-bench/t3-acfaf281')
plan=json.loads((E/'parent-source-shards-plan.json').read_text())
validation=json.loads((W/'results/diagnostics/minimax-m31-flash/validation.json').read_text())
report={'source_shards':[],'models':{},'drift':{},'method':'Complete pinned vision-containing source shards, LFS SHA256, direct safe_open source tensors with independent key mapping; no vtb source loader or range cache.'}
bundles={'minimax_m3':E/'m3-bundle','minimax_m31_flash':W/'hf/MiniMax-M3.1-Flash-Vision'}
checkpoint=E/'parent-source-shard-checkpoint.json'
verified=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
start=time.monotonic()
for row in plan:
    p=Path(hf_hub_download(row['repo'],row['file'],revision=row['revision']))
    prior=verified.get(row['sha256'])
    if prior and prior['size']==p.stat().st_size and prior['mtime_ns']==p.stat().st_mtime_ns:
        assert prior['sha256_pass']
    else:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
        assert h.hexdigest()==row['sha256']
        verified[row['sha256']]={'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256_pass':True}
        checkpoint.write_text(json.dumps(verified,indent=2)+'\n')
    assert p.stat().st_size==row['bytes']
    report['source_shards'].append({**row,'path':str(p),'sha256_pass':True})
    print('Verified source shard',row['repo'],row['file'],flush=True)
for model,repo,bundle in [('minimax_m3','MiniMaxAI/MiniMax-M3','MiniMax-M3-Vision'),('minimax_m31_flash','MiniMaxAI/MiniMax-M3.1-Flash','MiniMax-M3.1-Flash-Vision')]:
    matched={};counts={'tower':0,'projector':0}
    with safe_open(bundles[model]/'model.safetensors',framework='pt') as tower,safe_open(bundles[model]/'projector.safetensors',framework='pt') as projector:
        for row in report['source_shards']:
            if row['repo']!=repo:continue
            with safe_open(row['path'],framework='pt') as source:
                for name in source.keys():
                    if name.startswith('vision_tower.vision_model.'):
                        key=name.removeprefix('vision_tower.vision_model.').replace('encoder.layers.','layers.').replace('embeddings.patch_embedding.','embeddings.proj.');part='tower';target=tower
                    elif name.startswith('multi_modal_projector.'):
                        key=name.removeprefix('multi_modal_projector.');part='projector';target=projector
                    elif name.startswith('patch_merge_mlp.'):
                        key=name.removeprefix('patch_merge_mlp.').replace('linear_','merge_linear_');part='projector';target=projector
                    else:continue
                    assert (part,key) not in matched
                    a,b=source.get_tensor(name),target.get_tensor(key)
                    assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),(model,name,key)
                    digest=hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    assert validation['identity'][model]['model.safetensors' if part=='tower' else 'projector.safetensors'][key]['sha256']==digest
                    matched[(part,key)]=digest;counts[part]+=1
        assert counts=={'tower':515,'projector':8}
        assert counts['tower']==len(tower.keys()) and counts['projector']==len(projector.keys())
    report['models'][model]={'counts':counts,'all_bytes_and_dtypes_exact':True,'all_recorded_tensor_hashes_match':True}
for part,file in [('tower','model.safetensors'),('projector','projector.safetensors')]:
    expected={r['name']:r for r in validation['drift'][part]};unequal=0
    with safe_open(bundles['minimax_m3']/file,framework='pt') as first,safe_open(bundles['minimax_m31_flash']/file,framework='pt') as second:
        assert set(first.keys())==set(second.keys())==set(expected)
        for key in first.keys():
            a,b=first.get_tensor(key).double().flatten(),second.get_tensor(key).double().flatten()
            eq=torch.equal(a,b);cos=torch.nn.functional.cosine_similarity(a,b,dim=0).item();rel=((b-a).norm()/a.norm()).item()
            row=expected[key];assert eq==row['equal'] and abs(cos-row['cosine'])<1e-10 and abs(rel-row['relative_l2'])<1e-10,(part,key)
            unequal+=not eq
    report['drift'][part]={'tensors':len(expected),'unequal':unequal,'all_recorded_cosines_relative_l2_match':True}
report['seconds']=time.monotonic()-start;report['passed']=True
(E/'parent-source-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_shards'},indent=2),flush=True)
