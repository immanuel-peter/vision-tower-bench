"""Count bootstrap: prediction parsing, paired MAE/RMSE intervals, comparison families, CLI, table."""

import hashlib
import json
import math

import numpy as np
import pytest
import torch

from scripts import count_bootstrap as cli
from scripts import count_table
from vtb import count_bootstrap as cb
from vtb.counting import count_bin
from vtb.geometry_bootstrap import paired_metric_bootstrap

HEADER = "\t".join(cb.COLUMNS)


def tsv_rows(ids, true, predicted, stage, layer, arm, split="test", digits=4):
    """Contract rows; predicted is (seeds, images)."""
    rows = []
    for seed in range(predicted.shape[0]):
        for i, image_id in enumerate(ids):
            signed = predicted[seed, i] - true[i]
            rows.append("\t".join(str(v) for v in (
                image_id, split, stage, layer, arm, seed, int(true[i]), f"{predicted[seed, i]:.{digits}f}",
                f"{signed:.{digits}f}", f"{abs(signed):.{digits}f}", f"{signed * signed:.{digits}f}",
                count_bin(true[i]))))
    return rows


def cell_json(stage, layer, true, predicted):
    per_seed = [{"mae": round(float(np.abs(p - true).mean()), 4),
                 "rmse": round(float(np.sqrt(((p - true) ** 2).mean())), 4)} for p in predicted]
    mae = [s["mae"] for s in per_seed]
    return {"stage": stage, "layer_index": layer, "relative_depth": 1.0, "model_id": "toy/model",
            "learning_rate": 1e-3, "mae": float(np.mean(mae)), "mae_std": float(np.std(mae)),
            "rmse": float(np.mean([s["rmse"] for s in per_seed])), "per_seed": per_seed,
            "images": {"train": 10, "val": 5, "test": len(true)}, "token_width": 8, "input_width": 8,
            "grid": 4, "trainable_parameters": 100}


STAGE_NOISE = {"tower": 1.0, "merged": 2.0, "projected": 4.0}


def make_truth(images=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"fsc147_{k}" for k in range(images)]
    true = np.concatenate([rng.integers(1, 50, images - 12), rng.integers(51, 200, 8), rng.integers(201, 900, 4)])
    return ids, true.astype(np.float64)


def write_result(directory, model, arm, capacity, ids, true, stages, complete=True, seeds=3, offset=0.0,
                 rng_seed=1, val=True):
    """A runner-shaped result: payload JSON, its predictions TSV and the .cells directory."""
    rng = np.random.default_rng(rng_seed)
    stem = f"{model}_count_{arm}_{capacity}"
    cells_dir = directory / f"{stem}.cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cells, rows = [], []
    for stage in stages:
        layer = 27
        scale = STAGE_NOISE[stage] + offset
        predicted = true[None, :] + rng.normal(0, scale, (seeds, len(true))) * np.sqrt(true)[None, :] / 3
        cell = cell_json(stage, layer, true, predicted)
        key = f"{stage}_L{layer:02d}"
        lines = tsv_rows(ids, true, predicted, stage, layer, arm)
        (cells_dir / f"{key}.predictions.tsv").write_text(HEADER + "\n" + "\n".join(lines) + "\n")
        (cells_dir / f"{key}.json").write_text(json.dumps(cell))
        if val:
            vtrue = true[:10]
            vpred = vtrue[None, :] + rng.normal(0, 1.0, (seeds, 10))
            vids = [f"fsc147_v{k}" for k in range(10)]
            (cells_dir / f"{key}.val.tsv").write_text(
                HEADER + "\n" + "\n".join(tsv_rows(vids, vtrue, vpred, stage, layer, arm, split="val")) + "\n")
            if arm == "exemplar":
                cell["one_exemplar_val"] = {"note": "transfer diagnostic only; never used for selection",
                                            "mae": 3.5, "mae_std": 0.2, "rmse": 6.0, "rmse_std": 0.3,
                                            "images": 10, "per_seed": []}
                (cells_dir / f"{key}.json").write_text(json.dumps(cell))
        cells.append(cell)
        rows += lines
    payload = {"task": "count", "arm": arm, "capacity_matched": capacity == "matched",
               "expected_cells": len(stages), "complete": complete, "cells": cells}
    (directory / f"{stem}.json").write_text(json.dumps(payload))
    if complete:
        (directory / f"{stem}.predictions.tsv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return directory / f"{stem}.json"


@pytest.fixture
def matrix(tmp_path):
    ids, true = make_truth()
    eligible = [i for k, i in enumerate(ids) if k % 5 != 0]
    keep = np.array([k % 5 != 0 for k in range(len(ids))])
    for capacity, offset in (("raw", 0.5), ("matched", 0.0)):
        write_result(tmp_path, "toy", "free", capacity, ids, true, ("tower", "merged", "projected"), offset=offset,
                     rng_seed=1 + len(capacity))
        write_result(tmp_path, "toy", "exemplar", capacity, eligible, true[keep], ("tower",), offset=offset - 0.5,
                     rng_seed=7 + len(capacity))
    return tmp_path, ids, true, eligible


# ------------------------------------------------------------------ statistics
def test_draws_reproduce_paired_metric_bootstrap_exactly():
    torch.manual_seed(0)
    first, second = torch.rand(3, 41) * 5, torch.rand(3, 41) * 5 + 0.3
    reference = paired_metric_bootstrap(first, second, higher_is_better=False, resamples=1000, seed=0)
    diff = second.float().mean(0) - first.float().mean(0)
    dist = torch.cat([diff[index].mean(dim=1) for index in cb.draws(41, 1000, 0)])
    low, high = torch.quantile(dist, torch.tensor([0.025, 0.975]))
    assert (low.item(), high.item()) == (reference["lower"], reference["upper"])
    assert reference["difference"] == cb.DIFFERENCE


def test_rmse_bootstrap_point_estimate_orientation_and_pooled():
    torch.manual_seed(1)
    first = torch.rand(3, 50) * 4
    second = first * 4  # squared errors four times larger: RMSE doubles
    out = cb.paired_rmse_bootstrap(first, second, resamples=500)
    expected = cb.seed_mean_rmse(second) - cb.seed_mean_rmse(first)
    assert out["point_estimate"] == pytest.approx(expected)
    assert out["point_estimate"] == pytest.approx(cb.seed_mean_rmse(first))
    assert 0 < out["lower"] <= out["point_estimate"] <= out["upper"]
    assert out["pooled"]["point_estimate"] == pytest.approx(math.sqrt(second.double().mean()) - math.sqrt(first.double().mean()))
    assert cb.resolved(out) == "first"
    same = cb.paired_rmse_bootstrap(first, first.clone(), resamples=200)
    assert same["lower"] == same["upper"] == same["point_estimate"] == 0.0


def test_rmse_is_mean_of_per_seed_rmse_not_pooled_or_per_image():
    sq = torch.tensor([[1.0, 9.0], [100.0, 100.0]], dtype=torch.float64)
    assert cb.seed_mean_rmse(sq) == pytest.approx((math.sqrt(5.0) + 10.0) / 2)
    assert cb.pooled_rmse(sq) == pytest.approx(math.sqrt(52.5))


def test_seeds_are_not_extra_images():
    torch.manual_seed(2)
    first, second = torch.rand(1, 30), torch.rand(1, 30) + 0.2
    one = cb.compare(*cells_from(first, second), resamples=400)
    tripled = cb.compare(*cells_from(first.repeat(3, 1), second.repeat(3, 1)), resamples=400)
    for metric in ("mae", "rmse"):
        for key in ("point_estimate", "lower", "upper"):
            assert tripled[metric][key] == pytest.approx(one[metric][key], abs=1e-6)
    assert tripled["images"] == one["images"] == 30


def cells_from(first_abs, second_abs, true=None, ids=None):
    images = first_abs.shape[1]
    ids = ids or [f"i{k:03d}" for k in range(images)]
    true = np.full(images, 10.0) if true is None else true
    bins = [count_bin(t) for t in true]

    def cell(errors, stage):
        errors = errors.double()
        return cb.Cell(stage, 1, "free", "test", list(ids), list(range(errors.shape[0])), true,
                       torch.from_numpy(true)[None, :] + errors, errors, errors.abs(), errors.pow(2), bins)

    return cell(first_abs, "tower"), cell(second_abs, "projected")


def test_compare_requires_identical_ids_and_seeds():
    a, b = cells_from(torch.rand(3, 10), torch.rand(3, 10))
    with pytest.raises(ValueError, match="identical image IDs"):
        cb.compare(a, b.subset(b.image_ids[:9]), resamples=10)
    b.seeds = [0, 1, 3]
    with pytest.raises(ValueError, match="seeds"):
        cb.compare(a, b, resamples=10)


def test_per_bin_intervals_carry_counts_and_small_sample_flags():
    true = np.array([10.0] * 40 + [100.0] * 12 + [300.0] * 3)
    a, b = cells_from(torch.rand(3, 55), torch.rand(3, 55) + 1.0, true=true)
    out = cb.compare(a, b, resamples=300)
    assert out["by_bin"]["le50"]["images"] == 40 and not out["by_bin"]["le50"]["small_sample"]
    assert out["by_bin"]["50to200"]["images"] == 12 and out["by_bin"]["50to200"]["small_sample"]
    assert out["by_bin"]["gt200"]["images"] == 3 and out["by_bin"]["gt200"]["small_sample"]
    assert out["by_bin"]["gt200"]["mae"]["point_estimate"] == pytest.approx(
        (b.abs_error[:, 52:].mean() - a.abs_error[:, 52:].mean()).item(), abs=1e-5)
    assert out["mae"]["lower"] > 0 and out["by_bin"]["le50"]["first_summary"]["images"] == 40
    empty = cb.compare(*cells_from(torch.rand(3, 5), torch.rand(3, 5)), resamples=10)
    assert empty["by_bin"]["gt200"] == {"images": 0, "small_sample": True}


# ------------------------------------------------------------------ parsing and sources
def test_read_predictions_round_trip_and_validation(tmp_path):
    ids, true = make_truth(12)
    predicted = true[None, :] + np.arange(24).reshape(2, 12) / 10
    path = tmp_path / "p.tsv"
    lines = tsv_rows(ids, true, predicted, "tower", 5, "free")
    path.write_text(HEADER + "\n" + "\n".join(lines) + "\n")
    cell = cb.read_predictions(path)["tower_L05"]
    assert cell.image_ids == sorted(ids) and cell.seeds == [0, 1]
    order = [ids.index(i) for i in cell.image_ids]
    assert cell.abs_error.numpy() == pytest.approx(np.abs(predicted - true)[:, order], abs=1e-4)

    path.write_text(HEADER + "\n" + "\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="every seed"):
        cb.read_predictions(path)
    bad = lines[0].split("\t")
    bad[11] = "gt200"
    path.write_text(HEADER + "\n" + "\n".join(["\t".join(bad)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="bin"):
        cb.read_predictions(path)
    bad = lines[0].split("\t")
    bad[8] = "99.0000"
    path.write_text(HEADER + "\n" + "\n".join(["\t".join(bad)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="signed_error"):
        cb.read_predictions(path)


def test_load_fsc_complete_partial_and_mismatch(tmp_path):
    ids, true = make_truth(30)
    path = write_result(tmp_path, "toy", "free", "matched", ids, true, ("tower", "merged"))
    result = cb.load_fsc(path)
    assert result.status == "complete" and set(result.cells) == {"tower_L27", "merged_L27"}
    tsv = path.with_suffix(".predictions.tsv")
    assert result.sources[str(tsv)] == hashlib.sha256(tsv.read_bytes()).hexdigest()
    assert set(result.val) == {"tower_L27", "merged_L27"}

    partial = write_result(tmp_path, "toy", "free", "raw", ids, true, ("tower", "merged"), complete=False)
    (partial.with_suffix(".cells") / "merged_L27.predictions.tsv").unlink()
    loaded = cb.load_fsc(partial)
    assert loaded.status.startswith("partial") and set(loaded.cells) == {"tower_L27"}
    assert cb.load_fsc(partial, allow_partial=False) is None

    cell = json.loads((path.with_suffix(".cells") / "tower_L27.json").read_text())
    payload = json.loads(path.read_text())
    payload["cells"][0]["per_seed"][0]["mae"] += 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match the cell JSON"):
        cb.load_fsc(path)
    assert cell["stage"] == "tower"


def test_comparison_families_on_a_matrix(matrix):
    root, ids, true, eligible = matrix
    results = {(r.arm, r.capacity): r for r in cb.discover(root)}
    assert set(results) == {("free", "raw"), ("free", "matched"), ("exemplar", "raw"), ("exemplar", "matched")}
    stages = cb.stage_comparisons(results["free", "matched"], resamples=300)
    assert set(stages) == {"stage:free_matched:tower_vs_merged", "stage:free_matched:merged_vs_projected",
                           "stage:free_matched:tower_vs_projected"}
    projected = stages["stage:free_matched:tower_vs_projected"]
    assert projected["images"] == len(ids) and projected["mae"]["point_estimate"] > 0
    assert projected["image_ids_sha256"] == cb.sha256_ids(sorted(ids))

    arm = cb.arm_comparison(results["free", "matched"], results["exemplar", "matched"], resamples=300)
    entry = arm["arm:tower_matched:free_vs_exemplar"]
    assert entry["images"] == len(eligible) and entry["dropped_from_first"] == len(ids) - len(eligible)
    assert entry["image_ids"] == sorted(eligible)
    # The free side is summarised on the common IDs, not on its own full test split.
    free_common = results["free", "matched"].final("tower").subset(sorted(eligible))
    assert entry["first_summary"]["mae"] == pytest.approx(free_common.abs_error.mean().item())

    capacity = cb.capacity_comparisons(results["free", "raw"], results["free", "matched"], resamples=300)
    assert set(capacity) == {f"capacity:free_{s}:raw_vs_matched" for s in cb.STAGES}

    with pytest.raises(ValueError):
        cb.arm_comparison(results["exemplar", "matched"], results["free", "matched"])
    swapped = results["exemplar", "matched"]
    swapped.cells["tower_L27"].image_ids[0] = "fsc147_unknown"
    with pytest.raises(ValueError, match="not in the free arm"):
        cb.arm_comparison(results["free", "matched"], swapped)


def test_cli_writes_evidence_with_sources_and_protocol(matrix, tmp_path):
    root = matrix[0]
    out = tmp_path / "analysis"
    cli.main(["fsc", "--results", str(root), "--out", str(out), "--resamples", "200", "--label", "test",
              "--relative-to", str(root)])
    payload = json.loads((out / "fsc147" / "toy_count_bootstrap.json").read_text())
    assert payload["label"] == "test" and not payload["partial"]
    assert payload["protocol"]["resamples"] == 200 and payload["protocol"]["bootstrap_seed"] == 0
    for name, sha in payload["sources"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    assert len(payload["comparisons"]) == 3 * 2 + 2 + 3 + 1
    assert all(set(c["by_bin"]) == set(cb.BINS) for c in payload["comparisons"].values())
    lines = (out / "fsc147" / "count_intervals.tsv").read_text().splitlines()
    assert lines[0].split("\t") == list(cli.TSV_COLUMNS)
    assert len(lines) - 1 == len(payload["comparisons"]) * 4 * 3

    # Same inputs and seed give byte-identical evidence.
    again = tmp_path / "again"
    cli.main(["fsc", "--results", str(root), "--out", str(again), "--resamples", "200", "--label", "test",
              "--relative-to", str(root)])
    assert (again / "fsc147" / "toy_count_bootstrap.json").read_bytes() == \
        (out / "fsc147" / "toy_count_bootstrap.json").read_bytes()


def write_countqa(directory, model, arm, variant, ids, true, stages, seeds=(0, 1, 2), rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows, cells = [], []
    for stage in stages:
        predicted = true[None, :] + rng.normal(0, STAGE_NOISE[stage], (len(seeds), len(true)))
        rows += tsv_rows(ids, true, predicted, stage, 27, arm, split="countqa", digits=6)
        cells.append({"key": f"{stage}_L27", "stage": stage, "layer": 27, "model_id": "toy/model",
                      "relative_depth": 1.0,
                      "metrics": {"images": len(ids), "mae": float(np.abs(predicted - true).mean()),
                                  "rmse": float(np.sqrt(((predicted - true) ** 2).mean())),
                                  "exact_match": float((np.floor(predicted + 0.5) == true).mean()),
                                  "mae_by_seed": np.abs(predicted - true).mean(axis=1).tolist(),
                                  "by_bin": {}}})
    path = directory / f"{model}_count_{arm}_{variant}.json"
    path.write_text(json.dumps({"model": model, "arm": arm, "variant": variant, "cells": cells}))
    path.with_suffix(".predictions.tsv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_countqa_stage_and_arm_pairs(tmp_path):
    ids = [f"countqa_{k:04d}_abcd" for k in range(40)]
    true = np.random.default_rng(3).integers(1, 60, 40).astype(float)
    free = write_countqa(tmp_path, "toy", "free", "matched", ids, true, ("tower", "merged", "projected"))
    exemplar = write_countqa(tmp_path, "toy", "exemplar", "matched", ids[:30], true[:30], ("tower",), rng_seed=1)
    out = tmp_path / "analysis"
    cli.main(["countqa", str(free), str(exemplar), "--out", str(out), "--resamples", "200"])
    payload = json.loads((out / "countqa" / "toy_count_bootstrap.json").read_text())
    assert set(payload["comparisons"]) == {
        "stage:free_matched:tower_vs_merged", "stage:free_matched:merged_vs_projected",
        "stage:free_matched:tower_vs_projected", "arm:tower_matched:free_vs_exemplar"}
    assert payload["comparisons"]["arm:tower_matched:free_vs_exemplar"]["images"] == 30
    assert payload["comparisons"]["stage:free_matched:tower_vs_projected"]["first"]["dataset"] == "countqa"


def test_dense_selection_is_deterministic_and_documented(matrix):
    root = matrix[0]
    results = cb.discover(root)
    chosen = cb.select_dense(results, k=2)
    assert chosen["status"] == "selected" and len(chosen["image_ids"]) == 2
    assert chosen["results_used"] == ["toy_count_free_matched", "toy_count_free_raw"]
    gt200 = {i for i, b in zip(results[0].cells["tower_L27"].image_ids, results[0].cells["tower_L27"].bins)
             if b == "gt200"}
    assert set(chosen["image_ids"]) <= gt200
    scores = [s["median_relative_jump"] for s in chosen["scores"]]
    assert scores == sorted(scores, reverse=True)
    assert cb.select_dense(list(reversed(results)), k=2) == chosen
    assert cb.select_dense([], k=2)["status"].startswith("pending")


# ------------------------------------------------------------------ table
def test_table_rows_bins_and_one_exemplar_diagnostic(matrix, tmp_path):
    root = matrix[0]
    results = cb.discover(root)
    rows = count_table.fsc_rows(results)
    stages = {(r["arm"], r["capacity"], r["stage"]) for r in rows if r["bin"] == "all"}
    assert ("free", "matched", "projected") in stages and ("exemplar", "raw", "tower") in stages
    tower = next(r for r in rows if r["arm"] == "free" and r["capacity"] == "matched"
                 and r["stage"] == "tower" and r["bin"] == "all")
    cell = next(r for r in results if (r.arm, r.capacity) == ("free", "matched")).cells["tower_L27"]
    assert tower["mae"] == pytest.approx(cell.abs_error.mean().item())
    assert tower["rmse"] == pytest.approx(cb.seed_mean_rmse(cell.sq_error))
    gt200 = next(r for r in rows if r["stage"] == "tower" and r["bin"] == "gt200" and r["arm"] == "free")
    assert gt200["images"] == 4 and gt200["small_sample"]
    diagnostics = count_table.one_exemplar_rows(results)
    assert {d["capacity"] for d in diagnostics} == {"raw", "matched"}
    assert diagnostics[0]["one_exemplar_val_mae"] == 3.5 and diagnostics[0]["three_box_val_mae"] > 0

    out = tmp_path / "table"
    count_table.main(["--results", str(root), "--out", str(out), "--label", "pilot"])
    markdown = (out / "count_table.md").read_text()
    assert "pilot" in markdown.lower() and "diagnostic" in markdown.lower()
    assert "n < 30" in markdown
    tsv = (out / "count_table.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == list(count_table.TSV_COLUMNS)
