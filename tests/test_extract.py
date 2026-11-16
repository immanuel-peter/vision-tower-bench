"""Full-grid extraction must also finish for merged/projected-only adapters."""

import json
import sys

import pytest
import torch
from PIL import Image

from vtb import extract
from vtb.feature_batch import FeatureBatch


@pytest.mark.parametrize("model,batch_size,batches", [
    ("moonvit_v2", 1, [1, 1]), ("kimi_k31_pro", 1, [1, 1]), ("other", 16, [2]),
])
def test_extract_cli_chooses_a_working_default_batch(monkeypatch, tmp_path, model, batch_size, batches):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("first", "second"):
        Image.new("RGB", (28, 28)).save(images / f"{name}.png")
    observed = []

    class Adapter:
        num_layers = 1
        model_id = "test/batch-default"
        stages = ("tower",)

        def __init__(self, resolution, **kwargs):
            self.resolution = resolution

        def depth_points(self):
            return [1]

        def preprocess(self):
            return lambda image: torch.zeros(3, 28, 28)

        def collate(self, rows):
            return torch.stack(rows)

        def extract(self, inputs, ids):
            observed.append(len(ids))
            yield FeatureBatch(tokens=torch.zeros(len(ids), 4, 8), image_ids=ids,
                               model_id=self.model_id, stage="tower", layer_index=1,
                               num_layers=1, resolution=self.resolution)

    monkeypatch.setitem(extract.ADAPTERS, model, Adapter)
    monkeypatch.setattr(sys, "argv", ["extract", "--model", model, "--images", str(images),
                                    "--out", str(tmp_path / "features"), "--device", "cpu", "--workers", "0"])
    extract.main()
    summary = json.loads((tmp_path / f"features/{model}_448_full/summary.json").read_text())
    assert observed == batches
    assert summary["images"] == 2 and summary["batch_size"] == batch_size


@pytest.mark.parametrize("model", ["moonvit_v2", "kimi_k31_pro"])
def test_invalid_kimi_batch_fails_before_loading_weights(monkeypatch, tmp_path, capsys, model):
    def forbidden(**kwargs):
        pytest.fail("invalid batch must be rejected before loading the adapter")

    monkeypatch.setitem(extract.ADAPTERS, model, forbidden)
    monkeypatch.setattr(sys, "argv", ["extract", "--model", model, "--images", str(tmp_path / "absent"),
                                    "--device", "cpu", "--batch-size", "2"])
    with pytest.raises(SystemExit) as error:
        extract.main()
    assert error.value.code == 2
    assert "requires --batch-size 1" in capsys.readouterr().err


def test_full_grid_extraction_without_tower_reports_nonzero_tokens(monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "sample.png")

    class Adapter:
        num_layers = 32
        resolution = 32
        model_id = "test/projector-only"
        stages = ("merged", "projected")

        def __init__(self, **_kwargs):
            pass

        def depth_points(self):
            return [32]

        def preprocess(self):
            return lambda _image: torch.zeros(3, 32, 32)

        def collate(self, rows):
            return {"pixel_values": torch.stack(rows)}

        def extract(self, _inputs, ids):
            for stage in self.stages:
                yield FeatureBatch(tokens=torch.zeros(1, 4, 8), image_ids=ids,
                                   model_id=self.model_id, stage=stage, layer_index=32,
                                   num_layers=32, resolution=32)

    monkeypatch.setitem(extract.ADAPTERS, "test-projector-only", Adapter)
    monkeypatch.setattr(sys, "argv", [
        "extract", "--model", "test-projector-only", "--images", str(images),
        "--out", str(tmp_path / "features"), "--workers", "0", "--device", "cpu",
        "--resolution", "32",
    ])
    extract.main()
    summary = json.loads((tmp_path / "features/test-projector-only_32_full/summary.json").read_text())
    assert summary["tokens_per_image"] == 4
    assert summary["stages"] == ["merged", "projected"]


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
