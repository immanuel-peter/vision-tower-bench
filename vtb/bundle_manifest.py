"""Reproducible vision-bundle manifests, excluding only self and Hub storage rules."""
import hashlib
import json
import shutil
from pathlib import Path
from safetensors import safe_open

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def copy_assets(source, out):
    source,out=Path(source),Path(out)
    for name in ('README.md','NOTICE','LICENSE-CODE','projector.py','examples/inference.py'):
        src,dst=source/name,out/name
        if src.is_file() and src.resolve()!=dst.resolve():
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)

def write_manifest(out, source_repo, source_revision, source_shards, *, merger_in_model=False, weight_dir=None, extra=None, allow_mixed_source=False):
    out=Path(out);weight_dir=Path(weight_dir or out)
    counts={};weight_paths={};dtypes={}
    for part,name in [('tower','model.safetensors'),('projector','projector.safetensors')]:
        p=weight_dir/name
        if not p.exists():
            if part=='projector' and merger_in_model:continue
            raise ValueError(f'missing pinned exported weight file: {p}')
        with safe_open(p,framework='pt') as weights:
            keys=list(weights.keys())
            for k in keys:
                dtype=weights.get_slice(k).get_dtype()
                if dtype != 'BF16' and not (allow_mixed_source and dtype == 'F32'):raise ValueError('expected original BF16 source tensors')
                dtypes[dtype]=dtypes.get(dtype,0)+1
            if part=='tower' and merger_in_model:
                counts['projector']=sum(k.startswith('merger.') for k in keys)
                counts['tower']=len(keys)-counts['projector']
            else:counts[part]=len(keys)
        weight_paths[name]=p
    hashes={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'export_manifest.json','.gitattributes'} and '__pycache__' not in p.relative_to(out).parts and '.cache' not in p.relative_to(out).parts}
    hashes.update({name:sha256(path) for name,path in weight_paths.items()})
    payload={k:v for k,v in (extra or {}).items() if k not in {'files','files_sha256','hash_excludes','tower_tensors','projector_tensors','shards'}}
    payload.update(source_repo=source_repo,source_revision=source_revision,source_shards=sorted(set(source_shards)),tower_tensors=counts['tower'],projector_tensors=counts['projector'],dtype='bfloat16' if set(dtypes)=={'BF16'} else 'mixed',tensor_dtypes=dtypes,files_sha256=hashes,hash_excludes=['export_manifest.json'],hub_only_metadata=['.gitattributes'])
    (out/'export_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload
