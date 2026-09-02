import pytest
import torch

from vtb.correspondence import (
    dense_map,
    geometric_errors,
    paired_bootstrap,
    ratio_matches,
    recall,
    semantic_errors,
)


def test_dense_map_preserves_raster_order():
    tokens = torch.arange(12).reshape(4, 3)
    mapped = dense_map(tokens)

    assert mapped.shape == (3, 2, 2)
    assert mapped[:, 0, 1].tolist() == [3, 4, 5]


def test_ratio_matches_recovers_unique_identity_matches():
    features = torch.eye(4)
    matches = ratio_matches(features, features, 4)

    assert sorted(zip(matches.source.tolist(), matches.target.tolist())) == [
        (0, 0), (1, 1), (2, 2), (3, 3)
    ]


def test_geometric_identity_has_zero_error():
    features = torch.eye(4).reshape(4, 2, 2)
    depth = torch.ones(2, 2)
    intrinsics = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])

    error_3d, error_2d = geometric_errors(
        features,
        features,
        depth,
        depth,
        intrinsics,
        intrinsics,
        torch.eye(4),
        resolution=2,
        evaluation_side=2,
        num_correspondences=4,
    )

    assert error_3d.tolist() == pytest.approx([0.0] * 4)
    assert error_2d.tolist() == pytest.approx([0.0] * 4)


def test_semantic_correspondence_scores_exact_keypoints():
    features = torch.eye(4).reshape(4, 2, 2)
    keypoints = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
    errors = semantic_errors(
        features,
        features,
        keypoints,
        keypoints,
        torch.tensor([True, True]),
        1.0,
        resolution=2,
    )

    assert errors.tolist() == pytest.approx([0.0, 0.0])
    assert recall(errors, 0.1) == 1.0


def test_paired_correspondence_bootstrap_is_reproducible():
    first = torch.tensor([0.8, 0.6, 0.4])
    second = torch.tensor([0.5, 0.5, 0.5])

    result = paired_bootstrap(first, second, resamples=1_000, seed=7)
    repeated = paired_bootstrap(first, second, resamples=1_000, seed=7)

    assert result == repeated
    assert result["point_estimate"] == pytest.approx(0.1)


def test_correspondence_runner_keeps_full_patch_grid_defaults():
    from scripts.correspondence_run import parser

    args = parser().parse_args([
        "--model", "dinov2", "--dataset", "scannet", "--root", "data", "--out", "out.json"
    ])

    assert args.resolution == 448
    assert args.evaluation_side == 64
    assert args.num_correspondences == 1_000


def test_correspondence_bootstrap_defaults_to_projected_against_tower():
    from scripts.correspondence_bootstrap import parser

    args = parser().parse_args(["--run", "run.json", "--out", "out.json"])

    assert args.first_stage == "projected"
    assert args.second_stage == "tower"
    assert args.resamples == 10_000
