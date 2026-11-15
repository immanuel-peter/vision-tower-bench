import json

import pytest

from vtb import correspondence_bootstrap


def payload():
    rows = []
    for identity, first, second, count in [("a", 1.0, 0.0, 1), ("b", 0.5, 0.5, 2), ("empty", float("nan"), float("nan"), 0)]:
        rows.append({"pair_id": identity, "viewpoint": 0, "cells": {
            "projected:27": {"pck_0.1": first, "keypoints": count},
            "tower:27": {"pck_0.1": second, "keypoints": count},
        }})
    return {"dataset": "spair", "model": "qwen38_flash_next", "model_id": "test",
            "resolution": 448, "evaluation_side": 64, "num_correspondences": 1000,
            "cells": [{"stage": s, "layer_index": 27} for s in ["tower", "projected"]],
            "measurements": rows}


def test_bootstrap_retains_finite_intersection_and_explicit_empty_pair_reason(tmp_path, monkeypatch):
    source, out = tmp_path / "source.json", tmp_path / "out.json"
    source.write_text(json.dumps(payload()))
    monkeypatch.setattr("sys.argv", ["bootstrap", "--run", str(source), "--out", str(out)])
    correspondence_bootstrap.main()
    result = json.loads(out.read_text())
    assert result["selected_pairs"] == 3
    assert result["first_advantage"]["pairs"] == 2
    assert result["first_advantage"]["point_estimate"] == 0.5
    assert result["excluded_nonfinite_pair_ids"] == ["empty"]
    assert result["exclusions"] == [{"pair_id": "empty", "reason": "zero visible keypoints"}]
    assert [r["pair_id"] for r in result["measurements"]] == ["a", "b"]


def test_bootstrap_rejects_nonfinite_pck_on_a_nonempty_pair(tmp_path, monkeypatch):
    source, out = tmp_path / "source.json", tmp_path / "out.json"
    data = payload()
    data["measurements"][-1]["cells"]["tower:27"]["keypoints"] = 1
    source.write_text(json.dumps(data))
    monkeypatch.setattr("sys.argv", ["bootstrap", "--run", str(source), "--out", str(out)])
    with pytest.raises(ValueError, match="nonempty pair"):
        correspondence_bootstrap.main()

