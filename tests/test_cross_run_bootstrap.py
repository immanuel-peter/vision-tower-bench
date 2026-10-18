import pytest

from scripts.cross_run_bootstrap import correspondence, pair_rows, parser, perturbation

NAMES = ("first", "second")


def args(*extra):
    return parser().parse_args(["correspondence", "--first", "a", "--first-name", "first",
                                "--second", "b", "--second-name", "second", "--out", "o", *extra])


def test_pairing_uses_identity_instead_of_file_order():
    a = [{"id": "b", "value": 2}, {"id": "a", "value": 1}]
    b = [{"id": "a", "value": 10}, {"id": "b", "value": 20}]
    assert [(x["value"], y["value"]) for x, y in pair_rows(a, b, "id")] == [(1, 10), (2, 20)]


@pytest.mark.parametrize("a,b", [([{"id": "x"}], [{"id": "y"}]),
    ([{"id": "x"}, {"id": "x"}], [{"id": "x"}]), ([], [])])
def test_pairing_rejects_missing_duplicate_and_empty_ids(a, b):
    with pytest.raises(ValueError):
        pair_rows(a, b, "id")


def test_correspondence_pairs_each_runs_own_cell_and_keeps_nonfinite_ids():
    def payload(values, layer):
        return {"dataset": "navi", "evaluation_side": 64, "num_correspondences": 1000,
                "cells": [{"stage": "tower", "layer_index": layer}],
                "measurements": [{"pair_id": str(i), "cells": {f"tower:{layer}": {"recall_2cm": v}}}
                                 for i, v in enumerate(values)]}
    result = correspondence(payload([.8, .5, float("nan")], 24), payload([.5, .4, .2], 27), NAMES, args())
    assert result["metric"] == "recall_2cm"
    assert result["finite_pairs"] == 2
    assert result["excluded_nonfinite_pair_ids"] == ["2"]
    assert result["difference"]["point_estimate"] == pytest.approx(.2)
    assert result["measurements"][0] == {"pair_id": "0", "first": .8, "second": .5}


def test_correspondence_rejects_different_scoring():
    base = {"dataset": "navi", "evaluation_side": 64, "num_correspondences": 1000, "cells": [], "measurements": []}
    with pytest.raises(ValueError, match="evaluation_side"):
        correspondence(base, base | {"evaluation_side": 32}, NAMES, args())


def test_perturbation_interval_uses_seed_mean_correctness_on_paired_test_images():
    def payload(correct, order):
        return {"images": 2000, "protocol": {"seeds": 3}, "cells": [{
            "stage": "tower", "layer_index": 24, "measurements": {"occlusion_l1": [
                {"image_id": key, "clean": [1.0] * 3, "perturbed": [correct] * 3} for key in order
            ]}}]}
    result = perturbation(payload(1.0, ["a", "b"]), payload(0.0, ["b", "a"]), NAMES, args())
    interval = result["conditions"]["occlusion_l1"]
    assert interval["difference"]["point_estimate"] == 1
    assert interval["difference"]["lower"] == 1
    assert interval["difference"]["upper"] == 1
    assert interval["test_images"] == 2
    assert interval["measurements"][0] == {"image_id": "a", "first": [1.0] * 3, "second": [0.0] * 3,
                                           "first_clean": [1.0] * 3, "second_clean": [1.0] * 3}
