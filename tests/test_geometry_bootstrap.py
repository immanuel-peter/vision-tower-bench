from types import SimpleNamespace

import pytest
import torch

from vtb import geometry_bootstrap_run
from vtb.bootstrap import paired_metric_bootstrap


def test_geometry_bootstrap_runner_defaults_to_matched_protocol():
    from vtb.geometry_bootstrap_run import parser

    args = parser().parse_args(
        [
            "--first-run", "first", "--first-name", "a", "--first-stage", "projected",
            "--first-layer", "50", "--second-run", "second", "--second-name", "b",
            "--second-stage", "merged", "--second-layer", "50", "--targets", "targets.npz",
            "--task", "depth", "--out", "out.json",
        ]
    )

    assert args.arm == "matched"
    assert args.epochs == 10
    assert args.seeds == 3


def test_paired_geometry_bootstrap_uses_seed_means_and_is_reproducible():
    first = torch.tensor([[0.8, 0.6, 0.4], [0.6, 0.8, 0.4]])
    second = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    result = paired_metric_bootstrap(
        first, second, higher_is_better=True, resamples=1_000, seed=7
    )
    repeated = paired_metric_bootstrap(
        first, second, higher_is_better=True, resamples=1_000, seed=7
    )

    assert result == repeated
    assert result["point_estimate"] == pytest.approx(0.1)
    assert result["difference"] == "first_minus_second"


def test_lower_geometry_metric_reports_positive_first_advantage():
    first = torch.tensor([[18.0, 20.0], [20.0, 18.0]])
    second = torch.tensor([[24.0, 22.0], [22.0, 24.0]])

    result = paired_metric_bootstrap(
        first, second, higher_is_better=False, resamples=100, seed=0
    )

    assert result["point_estimate"] == pytest.approx(4.0)
    assert result["difference"] == "second_minus_first"


def test_geometry_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="same"):
        paired_metric_bootstrap(
            torch.ones(2, 3), torch.ones(3, 3), higher_is_better=True
        )


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
    monkeypatch.setattr(geometry_bootstrap_run, "train_cell", train)
    first = geometry_bootstrap_run.cached_train_cell(run, "tower", 32, args)
    second = geometry_bootstrap_run.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 1
    assert torch.equal(first["values"], second["values"])
    assert first["image_ids"] == second["image_ids"]
    targets.write_bytes(b"changed target identity")
    geometry_bootstrap_run.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 2
    shard.write_bytes(b"changed feature identity")
    geometry_bootstrap_run.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 3
    args.learning_rates = [1e-3]
    geometry_bootstrap_run.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 4
    monkeypatch.setattr(geometry_bootstrap_run, "CELL_CACHE_VERSION", geometry_bootstrap_run.CELL_CACHE_VERSION + 1)
    geometry_bootstrap_run.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 5
    args.cell_cache = None
    geometry_bootstrap_run.cached_train_cell(run, "tower", 32, args)
    assert len(calls) == 6
