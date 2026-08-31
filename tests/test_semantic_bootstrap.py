import pytest
import torch

from vtb.semantic_bootstrap import paired_image_bootstrap


def test_bootstrap_runner_defaults_to_the_headline_arm():
    from scripts.semantic_bootstrap import parser

    args = parser().parse_args(
        [
            "--first-run", "first", "--first-name", "a", "--first-layer", "1",
            "--second-run", "second", "--second-name", "b", "--second-layer", "2",
            "--labels", "labels.json", "--out", "out.json",
        ]
    )

    assert args.readout == "attention"
    assert args.arm == "matched"


def test_paired_bootstrap_uses_seed_mean_correctness_and_is_reproducible():
    first = torch.tensor([[True, True, False, True], [True, False, False, True]])
    second = torch.tensor([[True, False, False, False], [True, False, True, False]])

    result = paired_image_bootstrap(first, second, resamples=1_000, seed=7)
    repeated = paired_image_bootstrap(first, second, resamples=1_000, seed=7)

    assert result == repeated
    assert result["point_estimate"] == pytest.approx(0.25)
    assert result["lower"] <= result["point_estimate"] <= result["upper"]


def test_paired_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="same"):
        paired_image_bootstrap(torch.ones(2, 3), torch.ones(3, 3))
