"""Count readout: head, metrics, official split, PCA isolation, persistence and resumption."""

import json
import sys

import numpy as np
import pytest
import torch

from vtb import cache, counting, geometry, geometry_run
from vtb.count_metrics import bins_for, density_loss, evaluate_count, summarise
from vtb.feature_batch import FeatureBatch
from vtb.probe_run import fit_reducer


@pytest.mark.parametrize("side", [14, 16, 28, 32])
def test_count_head_predicts_non_negative_density_on_the_declared_grid(side):
    head = geometry.CountHead([24], hidden_dim=16)
    density = head([torch.randn(2, 24, side, side)])
    assert density.shape == (2, 1, counting.TARGET_GRID, counting.TARGET_GRID)
    assert (density >= 0).all()


def test_count_metrics_on_a_known_example():
    density = torch.zeros(3, 1, 64, 64)
    density[0, 0, 0, 0], density[1, 0, 5, 5], density[2, 0, 9, 9] = 12.0, 40.0, 260.0
    metrics = evaluate_count(density, torch.tensor([10.0, 50.0, 250.0]))
    assert metrics["signed_error"].tolist() == pytest.approx([2.0, -10.0, 10.0])
    bins = bins_for(metrics["true_count"].tolist())
    assert bins == ["le50", "le50", "gt200"]
    summary = summarise(metrics, bins)
    assert summary["mae"] == pytest.approx(22 / 3, abs=1e-4)
    # RMSE comes from the mean squared error, not from averaging absolute errors.
    assert summary["rmse"] == pytest.approx((204 / 3) ** 0.5, abs=1e-4)
    assert summary["bias"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["by_bin"]["le50"]["images"] == 2 and summary["by_bin"]["50to200"] == {"images": 0}
    assert summary["by_bin"]["gt200"]["mae"] == 10.0
    assert density_loss(density, density).item() == 0.0


def test_official_split_follows_the_manifest_not_the_cache_order():
    ids = ["fsc147_3", "fsc147_1", "fsc147_2", "fsc147_4"]
    splits = {"train": ["fsc147_1", "fsc147_4"], "val": ["fsc147_2"], "test": ["fsc147_3"]}
    split = geometry_run.official_split(ids, splits)
    assert split.train.tolist() == [1, 3] and split.val.tolist() == [2] and split.test.tolist() == [0]
    kept = geometry_run.official_split(ids, splits, keep={"fsc147_4", "fsc147_3"})
    assert kept.train.tolist() == [3] and kept.val.tolist() == [] and kept.test.tolist() == [0]
    with pytest.raises(ValueError):
        geometry_run.official_split(ids[:3], splits)


def test_pca_fits_on_training_rows_only():
    tokens = torch.randn(6, 4, 8)
    tokens[4:] += 1000.0
    split = geometry_run.official_split(
        [f"i{k}" for k in range(6)], {"train": ["i0", "i1", "i2", "i3"], "val": ["i4"], "test": ["i5"]}
    )
    reducer = fit_reducer(tokens, split, 3)
    assert torch.allclose(reducer.mean, tokens[:4].reshape(-1, 8).mean(0, keepdim=True), atol=1e-5)
    again = fit_reducer(tokens.to(torch.bfloat16), split, 3)
    assert again.basis.shape == (8, 3)


def write_fixture(root, side=8, width=6, images=24, seed=0):
    """A synthetic cache whose density is readable from the features, plus prep outputs."""
    rng = np.random.default_rng(seed)
    ids = [f"fsc147_{k}" for k in range(images)]
    names = ["train"] * 14 + ["val"] * 5 + ["test"] * 5
    entries, targets = {}, {}
    tokens = torch.zeros(images, side * side, width)
    for k, image_id in enumerate(ids):
        count = int(rng.integers(0, 6)) * (k % 4)
        points = rng.random((count, 2)) * 0.98
        targets[image_id] = counting.density_map(points)
        for x, y in points:
            tokens[k, int(y * side) * side + int(x * side), 0] += 1.0
        tokens[k, :, 1:] = torch.from_numpy(rng.normal(size=(side * side, width - 1))).float() * 0.1
        boxes = [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7], [0.2, 0.6, 0.4, 0.9]]
        entries[image_id] = {"count": count, "split": names[k], "boxes": boxes,
                             "box_valid": [True, True, k != 3], "exemplar_eligible": k != 3,
                             "bin": counting.count_bin(count)}
    splits = {name: [i for i, n in zip(ids, names) if n == name] for name in ("train", "val", "test")}
    data = root / "data"
    data.mkdir()
    np.savez(data / "fsc147_targets.npz", **targets)
    (data / "fsc147_manifest.json").write_text(json.dumps({"images": entries, "splits": splits}))
    run = root / "features" / "toy_448_full"
    run.mkdir(parents=True)
    order = rng.permutation(images)
    for stage in ("tower", "merged"):
        FeatureBatch(tokens=tokens[order].to(torch.bfloat16), image_ids=[ids[i] for i in order],
                     model_id="toy/model", stage=stage, layer_index=4, num_layers=4,
                     resolution=448).save(run / f"{stage}_L04_00000.safetensors")
    cache.write_completion(run, {"model": "toy", "slices": cache.validate(run, ids, [("tower", 4), ("merged", 4)])})
    return run, data / "fsc147_targets.npz", ids


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["geometry_run", *map(str, argv)])
    geometry_run.main()


def test_cache_validation_rejects_missing_rows(tmp_path):
    run, _, ids = write_fixture(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        cache.validate(run, ids + ["fsc147_extra"], [("tower", 4), ("merged", 4)])
    with pytest.raises(ValueError, match="slices"):
        cache.validate(run, ids, [("tower", 4)])
    assert cache.completion(run)["sha256"]


@pytest.mark.parametrize("arm,matched", [("free", False), ("exemplar", True)])
def test_count_run_saves_reloadable_readouts_and_resumes(tmp_path, monkeypatch, arm, matched):
    run, targets, ids = write_fixture(tmp_path)
    out = tmp_path / "out" / f"toy_count_{arm}.json"
    out.parent.mkdir()
    argv = ["--task", "count", "--run", run, "--targets", targets, "--out", out, "--arm", arm,
            "--name", "toy", "--readouts", tmp_path / "readouts", "--device", "cpu", "--epochs", "2",
            "--batch-size", "4", "--learning-rates", "1e-3", "1e-2", "--seeds", "2", "--width", "4"]
    if matched:
        argv.append("--match-capacity")
    if arm == "exemplar":
        argv += ["--stages", "tower", "--deepest-only"]
    run_main(monkeypatch, *argv)

    payload = json.loads(out.read_text())
    assert payload["complete"] and payload["split"] == "official"
    cell = payload["cells"][0]
    assert cell["images"]["test"] == 5 and len(cell["per_seed"]) == 2
    assert set(cell["curves"]) == {"search", "seed 0", "seed 1"}
    assert len(cell["curves"]["seed 0"]["train_loss"]) == 2
    if arm == "exemplar":
        assert cell["images"]["train"] == 13 and "one_exemplar_val" in cell
        assert cell["input_width"] == 4 + 1

    rows = out.with_suffix(".predictions.tsv").read_text().splitlines()
    assert rows[0].split("\t") == list(geometry_run.PREDICTION_COLUMNS)
    assert len(rows) - 1 == len(payload["cells"]) * 2 * 5

    # A reloaded readout reproduces the saved test predictions from the raw cache.
    capacity = "matched" if matched else "raw"
    directory = tmp_path / "readouts" / "toy" / f"{arm}_{capacity}" / "tower_L04"
    readout = geometry_run.load_readout(directory, seed=1, device="cpu")
    assert readout.conditioned == (arm == "exemplar") and (readout.reducer is not None) == matched
    batch = cache.load_batch(run, "tower", 4)
    manifest = json.loads(targets.with_name("fsc147_manifest.json").read_text())
    test_ids = manifest["splits"]["test"]
    rows_of = [batch.image_ids.index(i) for i in test_ids]
    boxes = [np.array(manifest["images"][i]["boxes"]) for i in test_ids] if readout.conditioned else None
    counts = counting.masked_count(readout.density(batch.tokens[rows_of], boxes))
    saved = {(r.split("\t")[0], r.split("\t")[5]): float(r.split("\t")[7])
             for r in rows[1:] if r.split("\t")[2] == "tower"}
    for image_id, value in zip(test_ids, counts.tolist()):
        assert value == pytest.approx(saved[(image_id, "1")], abs=2e-3)
    if readout.conditioned:
        single = readout.density(batch.tokens[rows_of[:1]], [np.array(manifest["images"][test_ids[0]]["boxes"][:1])])
        assert single.shape == (1, 1, 64, 64)
        with pytest.raises(ValueError):
            readout.density(batch.tokens[rows_of[:1]])

    # A second invocation finds every cell complete and retrains nothing.
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "readouts").rglob("*.safetensors")}
    run_main(monkeypatch, *argv)
    assert {p: p.stat().st_mtime_ns for p in (tmp_path / "readouts").rglob("*.safetensors")} == before
    assert json.loads(out.read_text())["cells"] == payload["cells"]


def test_count_targets_align_by_image_id(tmp_path):
    _, targets, ids = write_fixture(tmp_path)
    images = json.loads(targets.with_name("fsc147_manifest.json").read_text())["images"]
    shuffled = ids[::-1]
    density, counts = geometry_run.count_targets(targets, shuffled, images)
    assert counts.tolist() == [images[i]["count"] for i in shuffled]
    assert density.flatten(1).sum(1).double().tolist() == pytest.approx(counts.tolist(), abs=1e-4)
