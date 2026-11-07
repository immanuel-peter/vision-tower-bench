from copy import deepcopy

import pytest

from scripts.validate_kimi_results import SEMANTIC_RATES, ids_digest, validate_perturbation


def payload():
    ids = [f"image-{index}" for index in range(2000)]
    splits = dict(train=ids[:1400], val=ids[1400:1700], test=ids[1700:])
    conditions = ["identity"] + [f"{factor}_l{level}" for factor in ("scale", "occlusion", "motion_blur") for level in range(1, 5)]
    interval = dict(point_estimate=0., lower=0., upper=0., resamples=10000, bootstrap_seed=0, confidence=.95)
    cell = dict(model_id="source", layer_index=27, relative_depth=1., capacity_matched=True, readout="attention",
                seed_ids=[0, 1, 2], learning_rate_search_seed=0, sample_ids_sha256=ids_digest(ids),
                split_ids_sha256={key: ids_digest(values) for key, values in splits.items()},
                train_images=1400, validation_images=300, test_images=300,
                learning_rate=.001, validation_accuracy=.5, learning_rate_search={str(rate): .5 for rate in SEMANTIC_RATES},
                conditions=[dict(condition=name, accuracy=1., accuracy_by_seed=[1., 1., 1.], clean_minus_condition=deepcopy(interval)) for name in conditions],
                measurements={name: [dict(image_id=image, clean=[1., 1., 1.], perturbed=[1., 1., 1.]) for image in splits["test"]] for name in conditions[1:]})
    output = dict(images=2000, transform_manifest_sha256="manifest",
                  protocol=dict(readout="attention", classes=100, capacity_matched_width=512, epochs=20, seeds=3,
                                learning_rates=list(SEMANTIC_RATES), learning_rate_search_seed=0, retraining_seeds=[0, 1, 2],
                                device="cpu", torch_threads=4, batch_size=256, standardize_channels=False),
                  feature_provenance={name: {"details": dict(model_id="source", release_revision="pinned", resolution=448, pooled_to=4,
                      batch_size=1, dtype="bfloat16", device="cuda:0", physical_gpu=1, threads=8, workers=2, condition=name,
                      ordered_ids_sha256=ids_digest(ids))} for name in conditions},
                  cells=[deepcopy(cell) | {"stage": stage} for stage in ("tower", "merged", "projected")])
    return output, ids, conditions, splits


def validate(example):
    output, ids, conditions, splits = example
    return validate_perturbation(output, "source", ids, conditions, splits, "pinned", "manifest")


def test_perturbation_validator_checks_actual_paired_correctness():
    example = payload()
    assert validate(example)["cells"] == 39
    example[0]["cells"][0]["conditions"][1]["accuracy"] = .5
    with pytest.raises(ValueError, match="accuracy differs"):
        validate(example)


@pytest.mark.parametrize("mutation, message", [
    (lambda p: p["cells"][0]["measurements"]["scale_l1"][0].update(image_id="wrong"), "test IDs"),
    (lambda p: p["cells"][0]["measurements"]["scale_l1"][0].update(perturbed=[1., 1.]), "three binary"),
    (lambda p: p["cells"][0]["conditions"][1].update(accuracy=float("nan")), "nonfinite"),
    (lambda p: p["feature_provenance"]["identity"]["details"].update(physical_gpu=0), "producer"),
    (lambda p: p["protocol"].update(learning_rate_search_seed=1), "protocol differs"),
    (lambda p: p["cells"][0]["learning_rate_search"].pop("0.001"), "learning-rate grid"),
    (lambda p: p["cells"][0]["conditions"].pop(), "condition cells"),
])
def test_perturbation_validator_rejects_incomplete_or_incompatible_results(mutation, message):
    example = payload()
    mutation(example[0])
    with pytest.raises(ValueError, match=message):
        validate(example)
