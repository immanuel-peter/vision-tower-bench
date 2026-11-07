from copy import deepcopy

import pytest

from scripts.validate_kimi_results import validate_correspondence, validate_pairing


def example():
    stages = ["tower", "merged", "projected"]
    payload = dict(dataset="spair", pairs=2, resolution=448,
                   cells=[dict(stage=s, layer_index=27, **{"macro_pck_0.1": .5}) for s in stages],
                   measurements=[dict(pair_id=name, cells={f"{s}:27": {"pck_0.1": score, "keypoints": count} for s in stages})
                                 for name, score, count in [("empty", float("nan"), 0), ("nonempty", .5, 4)]])
    visibility = {"measurements": [dict(pair_id="empty", valid_keypoints=0), dict(pair_id="nonempty", valid_keypoints=4)]}
    return payload, visibility


def test_empty_source_pairs_remain_present_and_pairing_reports_union():
    payload, visibility = example()
    report = validate_correspondence(payload, ["empty", "nonempty"], visibility)
    assert report["selected_pairs"] == 2 and report["nonempty_pairs"] == 1
    assert len(report["exclusions"]) == 3
    paired = validate_pairing(payload, deepcopy(payload), "tower:27")
    assert paired["finite_ids"] == ["nonempty"]
    assert paired["exclusion_union"][0]["pair_id"] == "empty"
    assert paired["same_excluded_ids"]


def test_nonfinite_nonempty_pair_fails():
    payload, visibility = example()
    payload["measurements"][1]["cells"]["tower:27"]["pck_0.1"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite metric on nonempty"):
        validate_correspondence(payload, ["empty", "nonempty"], visibility)


def test_wrong_exclusion_count_and_missing_selection_are_rejected():
    payload, visibility = example()
    payload["measurements"][0]["cells"]["tower:27"]["keypoints"] = 1
    with pytest.raises(ValueError, match="count differs"):
        validate_correspondence(payload, ["empty", "nonempty"], visibility)
    with pytest.raises(ValueError, match="IDs/order"):
        validate_correspondence(payload, ["nonempty"], visibility)


def test_cache_validator_checks_actual_tensors_and_ids(tmp_path):
    import torch
    from vtb.feature_batch import FeatureBatch
    from scripts.validate_kimi_results import validate_cache
    widths = {"tower": 1024, "merged": 4096, "projected": 7168}
    slices = [("tower", layer) for layer in (3, 7, 10, 14, 17, 20, 24, 27)] + [("merged", 27), ("projected", 27)]
    for stage, layer in slices:
        batch = FeatureBatch(torch.zeros(1, 16, widths[stage], dtype=torch.bfloat16),
                             ["one"], "source", stage, layer, 27, 448, 4)
        batch.save(tmp_path / f"{stage}_L{layer:02d}_00000.safetensors")
    assert validate_cache(tmp_path, "source", ["one"], "semantic")["shards"] == 10
    with pytest.raises(ValueError, match="image IDs"):
        validate_cache(tmp_path, "source", ["wrong"], "semantic")
    batch.tokens[0, 0, 0] = float("nan")
    batch.save(tmp_path / "projected_L27_00000.safetensors")
    with pytest.raises(ValueError, match="nonfinite cached tensor"):
        validate_cache(tmp_path, "source", ["one"], "semantic")


def test_primary_validator_checks_search_seeds_split_and_capacity_evidence():
    from scripts.validate_kimi_results import SEMANTIC_RATES, ids_digest, validate_probe
    ids = [f"image-{i}" for i in range(13000)]
    splits = {"train": ids[:9100], "val": ids[9100:11050], "test": ids[11050:]}
    cell = dict(stage="tower", layer_index=27, relative_depth=1.0, model_id="model",
                sample_ids_sha256=ids_digest(ids), seeds=3, seed_ids=[0, 1, 2],
                learning_rate_search_seed=0, learning_rate=.001,
                learning_rate_search={str(rate): .5 for rate in SEMANTIC_RATES},
                split_ids_sha256={name: ids_digest(values) for name, values in splits.items()},
                test_accuracy_by_seed=[.5, .5, .5], test_accuracy=.5, label_fraction=1.0,
                standardize_channels=False, token_width=1024, trainable_parameters=51300,
                train_images=9100)
    payload = dict(images=13000, label_fraction=1.0, standardize_channels=False,
                   train_images=9100, full_train_images=9100, cells=[cell],
                   feature_provenance={"details": dict(model_id="model", resolution=448, batch_size=1,
                                                       dtype="bfloat16", pooled_to=4, device="cpu")},
                   protocol=dict(epochs=20, readout="mean", classes=100, capacity_matched=True,
                                 width=512, learning_rates=list(SEMANTIC_RATES), retraining_seeds=[0, 1, 2]))
    assert validate_probe(payload, "model", ids, "label-budget", "matched", "mean", expected_splits=splits)["passed"]
    cell["seed_ids"] = [0, 0, 0]
    with pytest.raises(ValueError, match="seed protocol"):
        validate_probe(payload, "model", ids, "label-budget", "matched", "mean", expected_splits=splits)
    cell["seed_ids"] = [0, 1, 2]
    cell["split_ids_sha256"]["test"] = "wrong"
    with pytest.raises(ValueError, match="split identities"):
        validate_probe(payload, "model", ids, "label-budget", "matched", "mean", expected_splits=splits)
