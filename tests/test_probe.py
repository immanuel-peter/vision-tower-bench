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
    # Token i sits at row i // 32, column i % 32.
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
