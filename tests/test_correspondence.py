import json

import pytest
import torch
from PIL import Image

from vtb import correspondence_bootstrap
from vtb.bootstrap import paired_bootstrap
from vtb.correspondence import (
    geometric_errors,
    ratio_matches,
    recall,
    semantic_errors,
    token_dense_map,
)


def test_dense_map_preserves_raster_order():
    tokens = torch.arange(12).reshape(4, 3)
    mapped = token_dense_map(tokens)

    assert mapped.shape == (3, 2, 2)
    assert mapped[:, 0, 1].tolist() == [3, 4, 5]


def test_ratio_matches_recovers_unique_identity_matches():
    features = torch.eye(4)
    matches = ratio_matches(features, features, 4)

    assert sorted(zip(matches.source.tolist(), matches.target.tolist())) == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
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


def test_semantic_correspondence_accepts_bfloat16_stage_features():
    features = torch.eye(4).reshape(4, 2, 2).bfloat16()
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


def test_paired_correspondence_bootstrap_is_reproducible():
    first = torch.tensor([0.8, 0.6, 0.4])
    second = torch.tensor([0.5, 0.5, 0.5])

    result = paired_bootstrap(first, second, resamples=1_000, seed=7)
    repeated = paired_bootstrap(first, second, resamples=1_000, seed=7)

    assert result == repeated
    assert result["point_estimate"] == pytest.approx(0.1)


def test_correspondence_runner_keeps_full_patch_grid_defaults():
    from vtb.correspondence_run import parser

    args = parser().parse_args(
        [
            "--model",
            "dinov2",
            "--dataset",
            "scannet",
            "--root",
            "data",
            "--out",
            "out.json",
        ]
    )

    assert args.resolution is None
    assert args.evaluation_side == 64
    assert args.num_correspondences == 1_000


def test_scannet_rgb_is_aligned_to_the_depth_and_intrinsics_frame():
    from vtb.correspondence_run import align_rgb_to_depth

    image = Image.new("RGB", (1296, 968))
    depth = torch.ones(480, 640)

    aligned = align_rgb_to_depth(image, depth)

    assert aligned.size == (640, 480)


def test_correspondence_bootstrap_defaults_to_projected_against_tower():
    from vtb.correspondence_bootstrap import parser

    args = parser().parse_args(["--run", "run.json", "--out", "out.json"])

    assert args.first_stage == "projected"
    assert args.second_stage == "tower"
    assert args.resamples == 10_000


def payload():
    rows = []
    for identity, first, second, count in [
        ("a", 1.0, 0.0, 1),
        ("b", 0.5, 0.5, 2),
        ("empty", float("nan"), float("nan"), 0),
    ]:
        rows.append(
            {
                "pair_id": identity,
                "viewpoint": 0,
                "cells": {
                    "projected:27": {"pck_0.1": first, "keypoints": count},
                    "tower:27": {"pck_0.1": second, "keypoints": count},
                },
            }
        )
    return {
        "dataset": "spair",
        "model": "qwen38_flash_next",
        "model_id": "test",
        "resolution": 448,
        "evaluation_side": 64,
        "num_correspondences": 1000,
        "cells": [{"stage": s, "layer_index": 27} for s in ["tower", "projected"]],
        "measurements": rows,
    }


def test_bootstrap_retains_finite_intersection_and_explicit_empty_pair_reason(
    tmp_path, monkeypatch
):
    source, out = tmp_path / "source.json", tmp_path / "out.json"
    source.write_text(json.dumps(payload()))
    monkeypatch.setattr(
        "sys.argv", ["bootstrap", "--run", str(source), "--out", str(out)]
    )
    correspondence_bootstrap.main()
    result = json.loads(out.read_text())
    assert result["selected_pairs"] == 3
    assert result["first_advantage"]["pairs"] == 2
    assert result["first_advantage"]["point_estimate"] == 0.5
    assert result["excluded_nonfinite_pair_ids"] == ["empty"]
    assert result["exclusions"] == [
        {"pair_id": "empty", "reason": "zero visible keypoints"}
    ]
    assert [r["pair_id"] for r in result["measurements"]] == ["a", "b"]


def test_bootstrap_rejects_nonfinite_pck_on_a_nonempty_pair(tmp_path, monkeypatch):
    source, out = tmp_path / "source.json", tmp_path / "out.json"
    data = payload()
    data["measurements"][-1]["cells"]["tower:27"]["keypoints"] = 1
    source.write_text(json.dumps(data))
    monkeypatch.setattr(
        "sys.argv", ["bootstrap", "--run", str(source), "--out", str(out)]
    )
    with pytest.raises(ValueError, match="nonempty pair"):
        correspondence_bootstrap.main()
