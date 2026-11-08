import importlib.util
import json
from pathlib import Path
import re

import pytest
import torch
from safetensors.torch import save_file
from vtb.bundle_manifest import write_manifest,sha256


def test_manifest_counts_real_tensor_components_and_storage_exclusions(tmp_path):
    save_file({'tower.weight':torch.ones(2,dtype=torch.bfloat16),'merger.weight':torch.zeros(3,dtype=torch.bfloat16)},tmp_path/'model.safetensors')
    (tmp_path/'README.md').write_text('fixture')
    (tmp_path/'.gitattributes').write_text('Hub storage')
    m=write_manifest(tmp_path,'parent','pinned',['shard-b','shard-a'],merger_in_model=True)
    assert (m['tower_tensors'],m['projector_tensors'])==(1,1)
    assert m['source_shards']==['shard-a','shard-b']
    assert m['hash_excludes']==['export_manifest.json'] and m['hub_only_metadata']==['.gitattributes']
    assert set(m['files_sha256'])=={'README.md','model.safetensors'}
    assert m['files_sha256']['README.md']==sha256(tmp_path/'README.md')


def test_maintained_new_bundle_inventory_and_all_tracked_file_hashes():
    root=Path(__file__).parents[1]
    expected={'MiniMax-M3.1-Flash-Vision':(515,8),'Qwen3.8-Flash-Next-Vision':(327,6),
              'PixtralViT-Large-4':(272,4),'MoonViT-K3.1-Pro':(165,3)}
    checked=0
    for name,counts in expected.items():
        bundle=root/'hf'/name
        if not bundle.is_dir():continue
        checked+=1
        m=json.loads((bundle/'export_manifest.json').read_text())
        assert (m['tower_tensors'],m['projector_tensors'])==counts
        assert isinstance(m['source_shards'],list) and m['source_shards']
        assert m['dtype'] in {'mixed','bfloat16'}
        for rel,h in m['files_sha256'].items():
            if rel.endswith('.safetensors'):continue
            assert (bundle/rel).is_file() and sha256(bundle/rel)==h
        assert 'examples/inference.py' in m['files_sha256']
        if name!='Qwen3.8-Flash-Next-Vision':assert 'projector.py' in m['files_sha256']
    assert checked


def test_relocated_sidecars_keep_their_original_bytes():
    root=Path(__file__).parents[1]
    path=root/'results/diagnostics/kimi-k31-pro-checkpoint/execution-sidecars/relocations.json'
    if not path.exists():pytest.skip('Kimi-only evidence')
    for old,record in json.loads(path.read_text()).items():
        assert not (root/old).exists()
        assert sha256(root/record['new'])==record['sha256']
