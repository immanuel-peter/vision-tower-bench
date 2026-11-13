import json
from pathlib import Path

import pytest

from scripts.fetch_results import absent
from vtb.probe_run import split_indices

SPLITS = "results/evidence/qwen38-flash-next/splits.json"


@pytest.mark.skipif(bool(absent([SPLITS])), reason="payloads not fetched: uv run python scripts/fetch_results.py")
@pytest.mark.parametrize("dataset,counts", [
    ("diode", (541, 115, 115)),
    ("kitti", (700, 150, 150)),
    ("imagenet100", (9100, 1950, 1950)),
    ("perturbation", (1400, 300, 300)),
])
def test_prescribed_splits_match_recorded_counts_and_every_image_id(dataset, counts):
    path = Path(__file__).parents[1] / SPLITS
    evidence = json.loads(path.read_text())
    recorded = evidence["datasets"][dataset]
    ids = recorded["all_ids"]
    split = split_indices(len(ids), seed=evidence["seed"])
    actual = [split.train.tolist(), split.val.tolist(), split.test.tolist()]
    assert tuple(map(len, actual)) == counts
    assert sorted(index for group in actual for index in group) == list(range(len(ids)))
    for indices, key in zip(actual, ["train_ids", "validation_ids", "test_ids"]):
        assert [ids[i] for i in indices] == recorded[key]
