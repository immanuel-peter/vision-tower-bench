import pytest
import torch

from vtb.geometry_bootstrap import paired_metric_bootstrap


def test_geometry_bootstrap_runner_defaults_to_matched_protocol():
    from scripts.geometry_bootstrap import parser

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
