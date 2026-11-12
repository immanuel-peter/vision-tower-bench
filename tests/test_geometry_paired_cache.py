from types import SimpleNamespace

import torch

from scripts import geometry_bootstrap


def test_paired_cell_cache_reuses_exact_measurements_and_invalidates_changed_inputs(tmp_path, monkeypatch):
    run = tmp_path / "features"
    run.mkdir()
    shard = run / "tower_L32_00000.safetensors"
    shard.write_bytes(b"first feature identity")
    targets = tmp_path / "targets.npz"
    targets.write_bytes(b"first target identity")
    args = SimpleNamespace(
        cell_cache=tmp_path / "paired", targets=targets, task="depth", arm="matched",
        head="multiscale", width=512, epochs=10, batch_size=8, seeds=3,
        scale_invariant=False, max_depth=300., learning_rates=[1e-4, 3e-4], device="cpu",
    )
    calls = []
    def train(*unused):
        calls.append(1)
        return {"values": torch.tensor([[1.], [2.], [3.]]), "image_ids": ["test_0"]}
    monkeypatch.setattr(geometry_bootstrap, "train_cell", train)
    first = geometry_bootstrap.cached_train_cell(run, "tower", 32, args)
    second = geometry_bootstrap.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 1
    assert torch.equal(first["values"], second["values"])
    assert first["image_ids"] == second["image_ids"]
    targets.write_bytes(b"changed target identity")
    geometry_bootstrap.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 2
    shard.write_bytes(b"changed feature identity")
    geometry_bootstrap.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 3
    args.learning_rates = [1e-3]
    geometry_bootstrap.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 4
    args.cell_cache = None
    geometry_bootstrap.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 5
