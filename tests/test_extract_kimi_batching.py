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
