import argparse
import json

import torch

from vtb import cache, probe, probe_run
from vtb.feature_batch import FeatureBatch


def test_cache_round_trips_tokens_and_image_ids(tmp_path):
    for shard in range(2):
        ids = [f"img{shard}{i}" for i in range(4)]
        FeatureBatch(
            tokens=torch.arange(4 * 16 * 8, dtype=torch.bfloat16).reshape(4, 16, 8) + shard,
            image_ids=ids,
            model_id="test/model",
            stage="tower",
            layer_index=7,
            num_layers=14,
            resolution=448,
            pooled_to=4,
        ).save(tmp_path / f"tower_L07_{shard:05d}.safetensors")

    assert cache.slices(tmp_path) == [("tower", 7)]
    tokens, image_ids, meta = cache.load(tmp_path, "tower", 7)
    assert tokens.shape == (8, 16, 8)
    assert image_ids == [f"img{s}{i}" for s in range(2) for i in range(4)]
    assert meta["relative_depth"] == "0.5000"


def test_reducer_keeps_a_linearly_separable_signal():
    torch.manual_seed(0)
    signal = torch.cat([torch.ones(64, 16, 1), -torch.ones(64, 16, 1)])
    tokens = torch.cat([signal, torch.randn(128, 16, 255) * 0.1], dim=-1)

    reduced = probe.Reducer.fit(tokens, 8)(tokens)
    assert reduced.shape == (128, 16, 8)
    means = reduced[:64].mean(dim=(0, 1)), reduced[64:].mean(dim=(0, 1))
    assert (means[0] - means[1]).abs().max() > 1.0


def test_capacity_matching_equalizes_trainable_parameters():
    counts = {probe.parameter_count(probe.build("attention", 512, 100)) for _ in range(3)}
    assert len(counts) == 1
    wide = probe.parameter_count(probe.build("attention", 7168, 100))
    narrow = probe.parameter_count(probe.build("attention", 1024, 100))
    assert wide > 3 * narrow


def test_splits_are_disjoint_and_cover_everything():
    split = probe_run.split_indices(1000)
    joined = torch.cat([split.train, split.val, split.test])
    assert len(joined) == 1000
    assert len(set(joined.tolist())) == 1000


def test_shard_writer_groups_batches_into_fixed_size_files(tmp_path):
    writer = cache.ShardWriter(tmp_path, images=4)
    for step in range(5):
        for layer in (3, 6):
            writer.add(
                FeatureBatch(
                    tokens=torch.full((2, 16, 8), float(step), dtype=torch.bfloat16),
                    image_ids=[f"img{step}a", f"img{step}b"],
                    model_id="test/model",
                    stage="tower",
                    layer_index=layer,
                    num_layers=12,
                    resolution=448,
                )
            )
    writer.close()

    assert len(list(tmp_path.glob("tower_L03_*.safetensors"))) == 3
    assert writer.slice_count == 2

    tokens, image_ids, _ = cache.load(tmp_path, "tower", 3)
    assert tokens.shape == (10, 16, 8)
    assert image_ids == [f"img{s}{c}" for s in range(5) for c in "ab"]


def test_dense_map_restores_the_patch_grid():
    from vtb import geometry

    batch = FeatureBatch(
        tokens=torch.randn(2, 1024, 64),
        image_ids=["a", "b"],
        model_id="test/model",
        stage="tower",
        layer_index=12,
        num_layers=24,
        resolution=448,
    )
    maps = geometry.dense_map(batch)
    assert maps.shape == (2, 64, 32, 32)
    # Token 101 maps to row 3, column 5.
    assert torch.equal(maps[0, :, 3, 5], batch.tokens[0, 3 * 32 + 5, :])


def test_dense_map_refuses_a_pooled_batch():
    import pytest

    from vtb import geometry

    batch = FeatureBatch(
        tokens=torch.randn(1, 1024, 8),
        image_ids=["a"],
        model_id="test/model",
        stage="tower",
        layer_index=12,
        num_layers=24,
        resolution=448,
    )
    with pytest.raises(ValueError, match="full patch tokens"):
        geometry.dense_map(batch.pooled(4))


def test_geometry_head_capacity_tracks_token_width():
    from vtb import geometry

    narrow = geometry.parameter_count(geometry.DepthHead([1024]))
    wide = geometry.parameter_count(geometry.DepthHead([7168]))
    assert wide > 2.5 * narrow


def test_depth_loss_is_only_partly_scale_invariant():
    from vtb.geometry_metrics import depth_si_loss

    target = torch.rand(2, 1, 32, 32) * 9 + 1
    assert depth_si_loss(target.clone(), target).item() == 0.0
    assert depth_si_loss(target * 2, target).item() > 2.0
    assert depth_si_loss(target * 2, target, lambda_scale=1.0).item() < 1e-6


def test_depth_metrics_reward_a_perfect_prediction():
    from vtb.geometry_metrics import evaluate_depth

    target = torch.rand(4, 64, 64) * 9 + 1
    metrics = evaluate_depth(target.clone(), target)
    assert metrics["rmse"].max().item() < 1e-6
    assert metrics["d1"].min().item() == 1.0


def test_scale_invariant_depth_recovers_a_scaled_prediction():
    from vtb.geometry_metrics import evaluate_depth

    target = torch.rand(4, 64, 64) * 9 + 1
    metrics = evaluate_depth(target * 3.0 + 2.0, target, scale_invariant=True)
    assert metrics["rmse"].max().item() < 1e-3


def test_geometry_runner_trains_a_depth_cell_end_to_end(tmp_path):
    """Runs the full path on synthetic data: shards in, trained head out, metrics back."""
    import numpy as np

    from vtb import cache, geometry, geometry_run
    from vtb.probe_run import split_indices

    torch.manual_seed(0)
    ids = [f"img{i:03d}" for i in range(24)]
    writer = cache.ShardWriter(tmp_path, images=8)
    writer.add(
        FeatureBatch(
            tokens=torch.randn(24, 64, 32),
            image_ids=ids,
            model_id="test/model",
            stage="tower",
            layer_index=6,
            num_layers=12,
            resolution=112,
        )
    )
    writer.close()

    store = {i: np.random.rand(16, 16).astype("float32") * 5 + 1 for i in ids}
    np.savez(tmp_path / "targets.npz", **store)

    batch = cache.load_batch(tmp_path, "tower", 6)
    assert batch.pooled_to is None
    assert batch.num_layers == 12
    targets, _ = geometry_run.load_targets(tmp_path / "targets.npz", batch.image_ids, "depth")
    assert targets.shape == (24, 1, 16, 16)

    features = geometry.dense_map(batch)
    assert features.shape == (24, 32, 8, 8)

    args = argparse.Namespace(
        head="linear", device="cpu", epochs=1, batch_size=4,
        learning_rate=1e-3, seed=0, scale_invariant=False, max_depth=10.0,
    )
    split = split_indices(24)
    model = geometry_run.train_cell(features, targets, None, split, "depth", args)
    metrics = geometry_run.score(model, features, targets, None, split.test, "depth", args)
    assert set(metrics) == {"d1", "d2", "d3", "rmse"}
    assert 0.0 <= metrics["d1"] <= 1.0


def test_depth_range_comes_from_the_manifest(tmp_path):
    import numpy as np

    from vtb.geometry_run import depth_range

    targets = tmp_path / "val_targets.npz"
    np.savez(targets, a=np.zeros((2, 2)))
    assert depth_range(targets, None) == 10.0

    (tmp_path / "val_manifest.json").write_text(json.dumps({"max_depth_metres": 230.64}))
    assert depth_range(targets, None) == 230.64
    assert depth_range(targets, 80.0) == 80.0


def test_depth_head_bins_span_the_requested_range():
    from vtb import geometry

    head = geometry.DepthHead([32], head="linear", max_depth=230.64)
    far = head([torch.randn(1, 32, 8, 8) * 50])
    assert head.predict.max_depth == 230.64
    # A head capped at NYU's 10 m cannot produce a DIODE-scale depth at all.
    near = geometry.DepthHead([32], head="linear", max_depth=10.0)
    assert near([torch.randn(1, 32, 8, 8) * 50]).max().item() <= 10.0
    assert far.shape == near([torch.randn(1, 32, 8, 8)]).shape
