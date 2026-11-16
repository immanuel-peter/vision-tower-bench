import importlib.util
import json
from pathlib import Path
import re

import pytest
import torch
from safetensors.torch import save_file
from vtb.bundle_manifest import write_manifest


def test_manifest_counts_real_tensor_components(tmp_path):
    save_file({'tower.weight':torch.ones(2,dtype=torch.bfloat16),'merger.weight':torch.zeros(3,dtype=torch.bfloat16)},tmp_path/'model.safetensors')
    m=write_manifest(tmp_path,'parent','pinned',['shard-b','shard-a'],merger_in_model=True)
    assert (m['tower_tensors'],m['projector_tensors'])==(1,1)
    assert m['source_shards']==['shard-a','shard-b']
    assert (m['source_repo'],m['source_revision'])==('parent','pinned')


def test_maintained_bundle_manifests_record_counts_and_shards():
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
    assert checked
