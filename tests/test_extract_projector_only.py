"""Full-grid extraction must also finish for merged/projected-only adapters."""

import json
import sys

import torch
from PIL import Image

from vtb import extract
from vtb.feature_batch import FeatureBatch


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
