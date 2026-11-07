import torch
import os
import pytest
from safetensors.torch import load_file
from PIL import Image

from vtb.adapters.minimax_m3 import MERGE_SIZE, PATCH_SIZE, Preprocess, merge_2x2
from vtb.adapters import minimax_m3 as mm


@pytest.mark.parametrize("checkpoint", [mm.M3, mm.FLASH])
def test_index_discovers_each_prefix_shards(monkeypatch, checkpoint):
    index = {}
    for prefix, count in [(mm.TOWER_PREFIX, 515), (mm.PROJECTOR_PREFIX, 4), (mm.MERGE_MLP_PREFIX, 4)]:
        for i in range(count):
            index[prefix + f"tensor_{i}"] = f"shard_{i % 3}"
    monkeypatch.setattr(mm, "source_json", lambda cp, filename: {"weight_map": index})
    def read(repo, shards, prefix, revision):
        assert repo == checkpoint.repo and revision == checkpoint.revision
        assert shards == sorted({v for k, v in index.items() if k.startswith(prefix)})
        return {k.removeprefix(prefix): torch.tensor(1) for k in index if k.startswith(prefix)}
    monkeypatch.setattr(mm, "load_prefixed", read)
    # Distinct names prevent the merge remapping from colliding with linear keys.
    monkeypatch.setattr(mm, "remap_projector", lambda a, b: {**a, **{"merge_" + k: v for k, v in b.items()}})
    tower, projector = mm.source_weights(checkpoint)
    assert len(tower) == 515 and len(projector) == 8


def test_incomplete_index_fetch_is_rejected(monkeypatch):
    monkeypatch.setattr(mm, "source_json", lambda *a: {"weight_map": {mm.TOWER_PREFIX + "x": "a"}})
    monkeypatch.setattr(mm, "load_prefixed", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="incomplete"):
        mm.source_weights(mm.FLASH)


def test_flash_rejects_m3_bundle_provenance(tmp_path):
    import json
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "export_manifest.json").write_text(json.dumps({
        "source_repo": mm.M3.repo, "source_revision": mm.M3.revision,
    }))
    with pytest.raises(ValueError, match="provenance"):
        mm.load_parts(torch.bfloat16, mm.FLASH, tmp_path)


def test_flash_requires_bundle_provenance(tmp_path):
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(ValueError, match="provenance"):
        mm.load_parts(torch.bfloat16, mm.FLASH, tmp_path)


@pytest.mark.skipif(bool(os.getenv("VTB_SKIP_WEIGHTS")), reason="weight checks disabled")
def test_flash_bundle_is_distinct_from_m3():
    for filename in ("model.safetensors", "projector.safetensors"):
        first = load_file(mm.M3.bundle / filename)
        second = load_file(mm.FLASH.bundle / filename)
        assert first.keys() == second.keys()
        assert any(not torch.equal(first[key], second[key]) for key in first)


def test_448_is_a_32_by_32_patch_grid():
    patches = Preprocess(448)(Image.new("RGB", (600, 480), "gray"))
    side = int(patches.shape[0] ** 0.5)
    assert side == 32
    assert side % MERGE_SIZE == 0
    assert patches.shape == (side * side, 3 * 2 * PATCH_SIZE * PATCH_SIZE)


def test_merge_2x2_groups_spatial_neighbors():
    rows, side, width = 1, 4, 3
    tokens = torch.arange(rows * side * side * width, dtype=torch.float32).view(rows, side * side, width)
    merged = merge_2x2(tokens)
    assert merged.shape == (1, 4, 12)
    # Top-left 2x2 of a 4x4 raster grid, flattened by position.
    top_left = tokens[0, [0, 1, 4, 5]].reshape(-1)
    assert torch.equal(merged[0, 0], top_left)
