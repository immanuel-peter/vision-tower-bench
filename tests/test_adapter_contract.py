"""Weight-free checks of the shared adapter helpers and the TowerAdapter contract."""

import json
import sys

import pytest
import torch
from PIL import Image
from torch import nn

from vtb import extract
from vtb.adapters import common, mimo26, nemotron35
from vtb.adapters.base import TowerAdapter
from vtb.adapters.common import PatchPreprocess, depth_points, make_batch, raster
from vtb.adapters.dinov3 import DINOv3Adapter
from vtb.feature_batch import STAGES, FeatureBatch

ADAPTER_CLASSES = sorted(set(extract.ADAPTERS.values()), key=lambda c: c.__name__)


def patch_ids(side: int, patch: int) -> torch.Tensor:
    ids = torch.arange(side * side, dtype=torch.float32).view(1, side, side)
    return ids.repeat_interleave(patch, 1).repeat_interleave(patch, 2).expand(3, -1, -1)


class Identity(PatchPreprocess):
    def pixels(self, image) -> torch.Tensor:
        return image


@pytest.mark.parametrize("patch,merge,temporal", [(14, 2, 2), (16, 2, 2), (14, 1, 1)])
def test_patch_preprocess_groups_by_merge_block_and_raster_restores_row_order(patch, merge, temporal):
    side = 8
    rows = Identity(side * patch, patch_size=patch, merge_size=merge, temporal_patch_size=temporal)(patch_ids(side, patch))
    assert rows.shape == (side * side, 3 * temporal * patch * patch)
    ids = rows[:, 0]
    if merge == 2:
        assert ids[:4].tolist() == [0.0, 1.0, side, side + 1.0]
        assert ids[4:8].tolist() == [2.0, 3.0, side + 2.0, side + 3.0]
    assert torch.equal(raster(ids.view(1, -1, 1), 1, merge).flatten(), torch.arange(side * side, dtype=torch.float32))


def test_raster_handles_batches_and_packed_rows_without_mixing_images():
    side, width = 6, 3
    grid = torch.arange(side * side).view(side // 2, 2, side // 2, 2).permute(0, 2, 1, 3).reshape(-1)
    grouped = torch.stack([grid + 100 * image for image in range(3)]).unsqueeze(-1).expand(-1, -1, width).float()
    expected = torch.stack([torch.arange(side * side) + 100 * image for image in range(3)]).float()
    assert torch.equal(raster(grouped, 3)[..., 0], expected)
    assert torch.equal(raster(grouped.reshape(-1, width), 3)[..., 0], expected)


@pytest.mark.parametrize("column_major", [False, True])
def test_mimo_packed_raster_matches_the_shared_raster_per_image(column_major):
    grid = torch.tensor([[1, 4, 4], [1, 6, 6]])
    tokens = torch.randn(16 + 36, 5)
    restored = mimo26.raster(tokens, grid, column_major)
    offset = 0
    for (_, h, w), actual in zip(grid.tolist(), restored, strict=True):
        sample = tokens[offset:offset + h * w]
        if column_major:
            sample = sample.reshape(w // 2, h // 2, 4, 5).transpose(0, 1)
        assert torch.equal(actual, raster(sample.reshape(1, -1, 5), 1)[0])
        offset += h * w


def test_make_batch_copies_adapter_fields_and_moves_tokens_to_cpu():
    adapter = type("Adapter", (), {"model_id": "org/tower", "num_layers": 24, "resolution": 448})()
    tokens = torch.randn(2, 4, 8)
    batch = make_batch(adapter, tokens, ["a", "b"], "merged", 24)
    assert isinstance(batch, FeatureBatch)
    assert (batch.model_id, batch.stage, batch.layer_index, batch.num_layers, batch.resolution) == ("org/tower", "merged", 24, 24, 448)
    assert batch.image_ids == ["a", "b"] and batch.tokens.device.type == "cpu"
    assert torch.equal(batch.tokens, tokens) and batch.relative_depth == 1.0


def test_depth_points_are_even_and_end_at_the_final_layer():
    assert depth_points(24) == [3, 6, 9, 12, 15, 18, 21, 24]
    assert depth_points(27) == [3, 7, 10, 14, 17, 20, 24, 27]
    assert depth_points(32, 4) == [8, 16, 24, 32]
    for n in (0, 25):
        with pytest.raises(ValueError, match="depth count"):
            depth_points(24, n)


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_default_device_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert common.default_device() == expected


@pytest.mark.parametrize("adapter_class", ADAPTER_CLASSES, ids=lambda c: c.__name__)
def test_every_adapter_declares_the_contract_and_collates_to_a_dict(adapter_class):
    assert adapter_class.default_resolution == (512 if adapter_class is DINOv3Adapter else 448)
    module = sys.modules[adapter_class.__module__]
    sample = module.Preprocess(adapter_class.default_resolution)(Image.new("RGB", (600, 480), "gray"))
    inputs = adapter_class.collate([sample, sample])
    assert isinstance(inputs, dict) and inputs and all(isinstance(v, torch.Tensor) for v in inputs.values())

    adapter = adapter_class.__new__(adapter_class)
    adapter.model_id, adapter.resolution, adapter.device, adapter.num_layers = "org/tower", 448, "cpu", 24
    adapter.selected_stages = STAGES
    assert isinstance(adapter, TowerAdapter)
    assert set(adapter.stages) <= set(STAGES) and adapter.stages[-1] in STAGES


class FakeRadio(nn.Module):
    def __init__(self):
        super().__init__()
        self.radio_model = nn.Module()
        self.radio_model.model = nn.Module()
        self.radio_model.model.blocks = nn.ModuleList(nn.Identity() for _ in range(32))


def test_nemotron35_stage_selection_is_per_instance_and_read_only(monkeypatch):
    monkeypatch.setattr(nemotron35, "load_parts", lambda *_: (FakeRadio(), nn.Identity(), nn.Identity()))
    bolt = nemotron35.Nemotron35BoltAdapter(device="cpu")
    tower_only = nemotron35.Nemotron35BoltAdapter(device="cpu", stages=["tower"])
    super_ = nemotron35.Nemotron35SuperAdapter(device="cpu")
    assert (bolt.stages, tower_only.stages, super_.stages) == (("merged", "projected"), ("tower",), STAGES)
    assert nemotron35.Nemotron35BoltAdapter.default_stages == ("merged", "projected")
    assert bolt.resolution == 448
    with pytest.raises(AttributeError):
        bolt.stages = ("tower",)
    with pytest.raises(ValueError, match="unknown or empty"):
        nemotron35.Nemotron35SuperAdapter(device="cpu", stages=["pooled"])


def test_extract_uses_the_adapter_default_resolution(monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "sample.png")
    seen = {}

    class Adapter:
        default_resolution = 512
        model_id = "test/default-resolution"
        stages = ("tower",)
        num_layers = 1

        def __init__(self, resolution, device):
            seen.update(resolution=resolution, device=device)
            self.resolution, self.device = resolution, device

        def depth_points(self):
            return [1]

        def preprocess(self):
            return lambda _image: torch.zeros(3, 4, 4)

        def collate(self, rows):
            return {"pixel_values": torch.stack(rows)}

        def extract(self, inputs, ids):
            yield FeatureBatch(tokens=torch.zeros(len(ids), 4, 8), image_ids=ids, model_id=self.model_id, stage="tower", layer_index=1, num_layers=1, resolution=self.resolution)

    monkeypatch.setitem(extract.ADAPTERS, "test-default-resolution", Adapter)
    monkeypatch.setattr(extract, "default_device", lambda: "cpu")
    monkeypatch.setattr(sys, "argv", ["extract", "--model", "test-default-resolution", "--images", str(images), "--out", str(tmp_path / "features"), "--workers", "0"])
    extract.main()
    assert seen == {"resolution": 512, "device": "cpu"}
    summary = json.loads((tmp_path / "features/test-default-resolution_512_full/summary.json").read_text())
    assert summary["resolution"] == 512 and summary["device"] == "cpu"
