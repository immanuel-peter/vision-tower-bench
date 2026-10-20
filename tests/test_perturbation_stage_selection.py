"""Regression for frozen-readout scoring without a Tower cache."""

import json
import sys

import pytest

from scripts import perturbation_run


def test_merged_projected_only_cache_reaches_scoring(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": 1, "conditions": ["identity"], "transforms": []}))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": {"sample": 0}}))
    monkeypatch.setattr(perturbation_run, "condition_details", lambda _: {"identity": {}})
    monkeypatch.setattr(perturbation_run.cache, "slices", lambda _: [("merged", 32), ("projected", 32)])
    visited = []

    def load(_path, stage, layer):
        visited.append((stage, layer))
        raise StopIteration("scoring reached")

    monkeypatch.setattr(perturbation_run.cache, "load", load)
    monkeypatch.setattr(sys, "argv", [
        "perturbation_run", "--clean-run", str(tmp_path), "--features-root", str(tmp_path),
        "--model-dir", "bolt", "--labels", str(labels), "--transform-manifest", str(manifest),
        "--out", str(tmp_path / "unused.json"),
    ])
    with pytest.raises(StopIteration, match="scoring reached"):
        perturbation_run.main()
    assert visited == [("merged", 32)]
