import json

import pytest
import torch

from scripts.extract_kimi_cpu import SLICES, WIDTHS, digest, restore_chunks, validate_batches
from vtb.feature_batch import FeatureBatch


def committed_chunk(root, provenance):
    chunk = root / ".chunks" / "00000"
    chunk.mkdir(parents=True)
    files = {}
    for stage, layer in SLICES:
        name = f"{stage}_L{layer:02d}_00000.safetensors"
        (chunk / name).write_bytes(f"verified-{stage}-{layer}".encode())
        files[name] = digest(chunk / name)
    (chunk / "chunk.json").write_text(json.dumps(dict(provenance=provenance, image_ids=["one"], files=files)))
    return chunk


def test_resume_restores_committed_chunks_and_rejects_corruption(tmp_path):
    provenance = {"device": "cpu", "threads": 4}
    chunk = committed_chunk(tmp_path, provenance)
    (tmp_path / ".staging-00001").mkdir()
    assert restore_chunks(tmp_path, ["one", "two"], provenance) == ["one"]
    assert (tmp_path / "tower_L03_00000.safetensors").stat().st_ino == (chunk / "tower_L03_00000.safetensors").stat().st_ino
    (chunk / "tower_L03_00000.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_chunks(tmp_path, ["one", "two"], provenance)


def test_resume_rejects_changed_device_or_order(tmp_path):
    committed_chunk(tmp_path, {"device": "cpu"})
    with pytest.raises(ValueError, match="provenance or ordered"):
        restore_chunks(tmp_path, ["one"], {"device": "cuda:0"})
    with pytest.raises(ValueError, match="provenance or ordered"):
        restore_chunks(tmp_path, ["wrong"], {"device": "cpu"})


def test_full_grid_validation_checks_finiteness_and_raster_order():
    batches = [FeatureBatch(torch.zeros(1, 1024 if stage == "tower" else 256, WIDTHS[stage], dtype=torch.bfloat16),
                            ["one"], "model", stage, layer, 27, 448) for stage, layer in SLICES]
    tower, merged = batches[7].tokens, batches[8].tokens
    tower[0, :, 0] = torch.arange(1024)
    merged.copy_(tower.reshape(1, 16, 2, 16, 2, 1024).permute(0, 1, 3, 2, 4, 5).reshape(1, 256, 4096))
    validate_batches(batches, "one")
    merged.copy_(merged.roll(1, dims=1))
    with pytest.raises(ValueError, match="raster-order"):
        validate_batches(batches, "one")
    batches[0].tokens[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite feature"):
        validate_batches(batches, "one")
