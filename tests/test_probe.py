import argparse
import json
import re
from pathlib import Path

import pytest
import torch

from vtb import cache, geometry_run, probe, probe_run
from vtb.feature_batch import FeatureBatch
from vtb.geometry_run import LEARNING_RATES


@pytest.mark.parametrize(
    ("script_name", "runner"),
    [("geometry_matrix.sh", geometry_run), ("semantic_matrix.sh", probe_run)],
)
def test_the_lane_script_searches_the_grid_the_runner_defines(script_name, runner):
    """One grid, defined once. A silent split between the two would truncate cells."""
    script = Path(__file__).parents[1] / "scripts" / script_name
    default = re.search(r'GRID="\$\{GRID:-([^}]*)\}"', script.read_text()).group(1)
    assert [float(rate) for rate in default.split()] == list(runner.LEARNING_RATES)


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


def test_semantic_runner_filters_cache_slices_by_depth_point(monkeypatch):
    slices = [("merged", 12), ("tower", 3), ("tower", 6), ("tower", 12)]
    monkeypatch.setattr(cache, "slices", lambda _run: slices)

    parser = probe_run.build_parser()
    args = parser.parse_args(
        ["--run", "cache", "--labels", "labels.json", "--depth-points", "3", "12"]
    )

    assert probe_run.selected_slices(args.run, args.depth_points) == [
        ("merged", 12),
        ("tower", 3),
        ("tower", 12),
    ]
    assert probe_run.selected_slices(args.run, None) == slices


def test_semantic_capacity_reducer_is_deterministic_and_preserves_rng_state():
    torch.manual_seed(7)
    features = torch.randn(32, 16, 24)
    split = probe_run.split_indices(len(features))

    torch.manual_seed(11)
    first = probe_run.fit_reducer(features, split, 8)
    after_first = torch.rand(4)
    torch.manual_seed(11)
    second = probe_run.fit_reducer(features, split, 8)
    after_second = torch.rand(4)

    assert torch.equal(first.basis, second.basis)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(after_first, after_second)


def test_geometry_capacity_reducer_is_deterministic_and_preserves_rng_state():
    torch.manual_seed(7)
    batch = FeatureBatch(
        tokens=torch.randn(32, 16, 24), image_ids=[str(i) for i in range(32)],
        model_id="test/model", stage="tower", layer_index=6, num_layers=12,
        resolution=56,
    )
    split = probe_run.split_indices(len(batch.image_ids))

    torch.manual_seed(11)
    first = geometry_run.match_capacity(batch, split, 8)
    after_first = torch.rand(4)
    torch.manual_seed(11)
    second = geometry_run.match_capacity(batch, split, 8)
    after_second = torch.rand(4)

    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(after_first, after_second)


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
        learning_rates=[1e-3, 3e-3], seeds=2, scale_invariant=False, max_depth=10.0,
    )
    split = split_indices(24)
    model = geometry_run.train_cell(features, targets, None, split, "depth", args, 1e-3, 0)
    metrics = geometry_run.score(model, features, targets, None, split.test, "depth", args)
    assert set(metrics) == {"d1", "d2", "d3", "rmse"}
    assert metrics["d1"].shape == (len(split.test),)

    coverage = geometry_run.coverage_of(targets, None)
    scenes = {i: "indoors" if n % 2 else "outdoor" for n, i in enumerate(ids)}
    summary = geometry_run.summarise(metrics, coverage, ids, split.test, scenes)
    assert 0.0 <= summary["d1"] <= 1.0
    assert summary["images"] == len(split.test)
    assert sum(b["images"] for b in summary["by_scene"].values()) == len(split.test)

    cell = geometry_run.run_cell(
        features, targets, None, split, "depth", coverage, ids, scenes, args
    )
    assert cell["learning_rate"] in args.learning_rates
    assert cell["seeds"] == 2 and len(cell["per_seed"]) == 2
    assert set(cell["learning_rate_search"]) == {"0.001", "0.003"}
    assert "d1_std" in cell and "d1_std" in cell["by_scene"]["indoors"]
    assert "coverage_std" not in cell


def test_learning_rate_selection_runs_the_right_way_per_task():
    from vtb import geometry_run

    assert geometry_run.SELECTION["depth"] == ("d1", True)
    assert geometry_run.SELECTION["normal"] == ("mean_deg", False)

    scored = {}

    def fake_score(model, features, targets, valid, index, task, args):
        rate = model
        return {geometry_run.SELECTION[task][0]: torch.tensor([scored[rate]])}

    def fake_train(features, targets, valid, split, task, args, learning_rate, seed):
        return learning_rate

    args = argparse.Namespace(learning_rates=[1e-4, 1e-3, 1e-2])
    split = argparse.Namespace(val=torch.arange(2))
    original = geometry_run.score, geometry_run.train_cell
    geometry_run.score, geometry_run.train_cell = fake_score, fake_train
    try:
        scored = {1e-4: 0.10, 1e-3: 0.90, 1e-2: 0.50}
        rate, report = geometry_run.select_learning_rate(None, None, None, split, "depth", args)
        assert rate == 1e-3 and report["val_score"] == 0.9

        scored = {1e-4: 30.0, 1e-3: 25.0, 1e-2: 40.0}
        rate, report = geometry_run.select_learning_rate(None, None, None, split, "normal", args)
        assert rate == 1e-3 and report["val_score"] == 25.0
        assert report["val_metric"] == "mean_deg"
    finally:
        geometry_run.score, geometry_run.train_cell = original


def test_aggregate_reports_mean_and_population_deviation():
    from vtb import geometry_run

    runs = [
        {"d1": 0.2, "images": 4, "coverage": 0.5, "by_scene": {"indoors": {"d1": 0.1, "images": 2, "coverage": 0.5}}},
        {"d1": 0.4, "images": 4, "coverage": 0.5, "by_scene": {"indoors": {"d1": 0.3, "images": 2, "coverage": 0.5}}},
    ]
    out = geometry_run.aggregate(runs)
    assert out["d1"] == 0.3 and out["d1_std"] == 0.1
    assert out["images"] == 4 and out["coverage"] == 0.5
    assert out["by_scene"]["indoors"]["d1"] == 0.2


def test_scene_summary_reads_coverage_for_the_images_it_scored():
    from vtb import geometry_run

    ids = [f"img{i:02d}" for i in range(10)]
    scenes = {i: "indoors" if n < 5 else "outdoor" for n, i in enumerate(ids)}
    coverage = torch.tensor([1.0] * 5 + [0.2] * 5)
    index = torch.tensor([7, 1, 9])
    metrics = {"d1": torch.tensor([0.3, 0.9, 0.5])}

    summary = geometry_run.summarise(metrics, coverage, ids, index, scenes)
    assert summary["by_scene"]["indoors"]["coverage"] == 1.0
    assert summary["by_scene"]["outdoor"]["coverage"] == 0.2
    assert summary["by_scene"]["indoors"]["d1"] == 0.9
    assert round(summary["coverage"], 4) == round((0.2 + 1.0 + 0.2) / 3, 4)


def test_bfloat16_cache_reaches_a_float32_head(tmp_path):
    from vtb import cache, geometry

    writer = cache.ShardWriter(tmp_path, images=2)
    writer.add(
        FeatureBatch(
            tokens=torch.randn(2, 64, 32).bfloat16(),
            image_ids=["a", "b"],
            model_id="test/model",
            stage="tower",
            layer_index=6,
            num_layers=12,
            resolution=112,
        )
    )
    writer.close()

    raw = geometry.dense_map(cache.load_batch(tmp_path, "tower", 6))
    assert raw.dtype == torch.bfloat16
    head = geometry.SurfaceNormalHead([32], head="linear")
    with pytest.raises(RuntimeError):
        head([raw])
    assert head([raw.float()]).shape == (2, 3, 32, 32)


def test_targets_are_cropped_to_the_square_the_tower_saw(tmp_path):
    import numpy as np

    from vtb import geometry_run

    depth = np.zeros((768, 1024), dtype="float32")
    depth[:, 128:896] = 1.0
    normal = np.tile(depth, (3, 1, 1)).astype("float16")
    np.savez(
        tmp_path / "val_targets.npz",
        a=depth, a_normal=normal, a_valid=depth.astype("uint8"),
    )

    cropped, valid = geometry_run.load_targets(tmp_path / "val_targets.npz", ["a"], "depth")
    assert cropped.shape == (1, 1, 768, 768)
    assert valid is None
    assert cropped.min().item() == 1.0

    normals, valid = geometry_run.load_targets(tmp_path / "val_targets.npz", ["a"], "normal")
    assert normals.shape == (1, 3, 768, 768)
    assert valid.shape == (1, 1, 768, 768)
    assert geometry_run.coverage_of(normals, valid).item() == 1.0


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
    near = geometry.DepthHead([32], head="linear", max_depth=10.0)
    assert near([torch.randn(1, 32, 8, 8) * 50]).max().item() <= 10.0
    assert far.shape == near([torch.randn(1, 32, 8, 8)]).shape


def test_shard_cache_key_tracks_the_revision_it_read(tmp_path, monkeypatch):
    from vtb import shards

    monkeypatch.setenv("VTB_SHARD_CACHE", str(tmp_path))
    assert shards.cache_root() == tmp_path

    first = shards.cache_path("org/model", "vision.", [("a.safetensors", "etag1")])
    assert first == shards.cache_path("org/model", "vision.", [("a.safetensors", "etag1")])
    assert first != shards.cache_path("org/model", "vision.", [("a.safetensors", "etag2")])
    assert first != shards.cache_path("org/model", "text.", [("a.safetensors", "etag1")])
    assert first != shards.cache_path("other/model", "vision.", [("a.safetensors", "etag1")])
