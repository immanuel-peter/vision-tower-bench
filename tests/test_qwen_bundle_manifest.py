import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from scripts import export_qwen3_8_vision as exporter
from vtb.adapters.qwen3_5 import FLASH_NEXT


def test_manifest_covers_nested_assets_and_only_source_vision_shards(monkeypatch, tmp_path):
    state = {f"tower.{i}": torch.ones(1, dtype=torch.bfloat16) for i in range(327)}
    state.update({f"merger.{i}": torch.ones(1, dtype=torch.bfloat16) for i in range(6)})
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/inference.py").write_text("print(1)\n")
    (tmp_path / "README.md").write_text("card\n")
    (tmp_path / "export_manifest.json").write_text("old manifest\n")
    monkeypatch.setattr(exporter, "source_json", lambda checkpoint, name: {"weight_map": {
        "model.visual.tower.weight": "vision.safetensors",
        "model.language.weight": "language.safetensors",
    }})
    exporter.write_export_manifest(tmp_path, FLASH_NEXT)
    manifest = json.loads((tmp_path / "export_manifest.json").read_text())
    assert manifest["source_shards"] == ["vision.safetensors"]
    assert (manifest["tower_tensors"], manifest["projector_tensors"]) == (327, 6)
    assert manifest["dtype"] == "bfloat16"
    assert manifest["hash_excludes"] == ["export_manifest.json"]
    expected = {"model.safetensors", "README.md", "examples/inference.py"}
    assert set(manifest["files_sha256"]) == expected
    for name in expected:
        assert manifest["files_sha256"][name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()


def test_manifest_rejects_non_bf16_weights(monkeypatch, tmp_path):
    save_file({f"tensor{i}": torch.ones(1) for i in range(333)}, tmp_path / "model.safetensors")
    monkeypatch.setattr(exporter, "source_json", lambda checkpoint, name: {"weight_map": {}})
    with pytest.raises(ValueError, match="BF16"):
        exporter.write_export_manifest(tmp_path, FLASH_NEXT)
