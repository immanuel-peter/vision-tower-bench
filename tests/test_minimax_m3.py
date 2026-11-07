import torch
import os
import pytest
from safetensors.torch import load_file
from PIL import Image

from vtb.adapters.minimax_m3 import MERGE_SIZE, PATCH_SIZE, Preprocess, merge_2x2
from vtb.adapters import minimax_m3 as mm
from types import SimpleNamespace


@pytest.mark.parametrize("cls", [mm.MiniMaxM3Adapter, mm.MiniMaxM31FlashAdapter])
def test_unrelated_images_cannot_change_another_images_features(cls):
    class MixingLayer(torch.nn.Module):
        def forward(self, tokens):
            return tokens + tokens.mean(dim=1, keepdim=True)
    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([MixingLayer() for _ in range(8)])
            self.config = SimpleNamespace(hidden_size=4)
        def forward(self, pixels, grid):
            tokens = pixels.unsqueeze(0)
            for layer in self.layers:
                tokens = layer(tokens)
            return SimpleNamespace(last_hidden_state=tokens)
    class Projector(torch.nn.Module):
        def forward(self, tokens):
            return tokens.reshape(-1, 4, 4).sum(dim=1)
    adapter = object.__new__(cls)
    adapter.model, adapter.projector = Tower(), Projector()
    adapter.num_layers, adapter.resolution = 8, 28
    adapter.device, adapter.dtype = "cpu", torch.float32
    adapter.model_id = cls.checkpoint.model_id
    first, second = torch.ones(4, 4), torch.full((4, 4), 10.)
    alone = list(adapter.extract(mm.collate([first]), ["first"]))
    paired = list(adapter.extract(mm.collate([first, second]), ["first", "second"]))
    reversed_pair = list(adapter.extract(mm.collate([second, first]), ["second", "first"]))
    assert len(alone) == len(paired) == 10
    for single, batch, reversed_batch in zip(alone, paired, reversed_pair):
        assert torch.equal(single.tokens[0], batch.tokens[0])
        assert torch.equal(single.tokens[0], reversed_batch.tokens[1])
        assert batch.image_ids == ["first", "second"]


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
