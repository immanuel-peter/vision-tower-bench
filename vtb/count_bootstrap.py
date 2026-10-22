"""Paired image bootstrap of count errors from saved prediction rows.

The resampling unit is the test image. Each comparison pairs two cells on
identical image IDs and identical seeds. Seeds are never extra images:

- MAE: each image's absolute error is averaged over seeds first, and the
  difference of those seed means is resampled with
  ``vtb.geometry_bootstrap.paired_metric_bootstrap``.
- RMSE: nonlinear, so every resample recomputes each seed's RMSE on the drawn
  images, averages the per-seed RMSEs, and takes the difference. The pooled
  RMSE (square root of the mean squared error over all seed x image rows) is
  reported beside it from the same draws.

Both metrics use the same image draws (same generator, seed and batching as
``paired_metric_bootstrap``). Differences are second minus first, so a positive
value means the second cell has the larger error. Intervals are percentile
intervals. Per-bin intervals resample within the bin and carry its image count.
"""

import csv
import hashlib
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from vtb.counting import BINS, count_bin
from vtb.geometry_bootstrap import paired_metric_bootstrap

COLUMNS = (
    "image_id", "split", "stage", "layer", "arm", "seed", "true_count",
    "predicted_count", "signed_error", "abs_error", "sq_error", "bin",
)
STAGES = ("tower", "merged", "projected")
STAGE_PAIRS = (("tower", "merged"), ("merged", "projected"), ("tower", "projected"))
SMALL_BIN = 30
RESAMPLES = 10_000
CONFIDENCE = 0.95
SEED = 0
BATCH = 256
RESULT_NAME = re.compile(r"^(?P<model>.+)_count_(?P<arm>free|exemplar)_(?P<capacity>raw|matched)$")
DIFFERENCE = "second_minus_first"
ORIENTATION = "second minus first; positive means the second cell has the larger error (first is better)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(image_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(image_ids).encode()).hexdigest()


# ------------------------------------------------------------------ prediction rows
@dataclass
class Cell:
    """One Stage cell's per-image, per-seed errors, image IDs sorted."""

    stage: str
    layer: int
    arm: str
    split: str
    image_ids: list[str]
    seeds: list[int]
    true: np.ndarray            # (N,)
    predicted: torch.Tensor     # (S, N) float64
    signed: torch.Tensor        # (S, N)
    abs_error: torch.Tensor     # (S, N)
    sq_error: torch.Tensor      # (S, N)
    bins: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.stage}_L{self.layer:02d}"

    def subset(self, image_ids: list[str]) -> "Cell":
        position = {image_id: k for k, image_id in enumerate(self.image_ids)}
        missing = [i for i in image_ids if i not in position]
        if missing:
            raise ValueError(f"{self.key}: {len(missing)} requested IDs are missing, e.g. {missing[:3]}")
        index = torch.tensor([position[i] for i in image_ids], dtype=torch.long)
        return Cell(self.stage, self.layer, self.arm, self.split, list(image_ids), list(self.seeds),
                    self.true[index.numpy()], self.predicted[:, index], self.signed[:, index],
                    self.abs_error[:, index], self.sq_error[:, index],
                    [self.bins[k] for k in index.tolist()], dict(self.meta))


def read_predictions(path: Path, split: str | None = None) -> dict[str, Cell]:
    """Parse a contract predictions TSV into cells keyed ``<stage>_L<layer>``.

    Every image must carry every seed exactly once with one true count, the bin
    must match ``count_bin`` of the true count, and the error columns must agree
    with predicted minus true to their printed precision.
    """
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if tuple(header or ()) != COLUMNS:
            raise ValueError(f"{path}: columns {header} are not the contract columns")
        grouped: dict[tuple, dict] = {}
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(COLUMNS):
                raise ValueError(f"{path}:{line}: {len(row)} fields")
            record = dict(zip(COLUMNS, row))
            if split is not None and record["split"] != split:
                continue
            group = grouped.setdefault((record["stage"], int(record["layer"]), record["arm"], record["split"]), {})
            slot = (record["image_id"], int(record["seed"]))
            if slot in group:
                raise ValueError(f"{path}:{line}: repeated image and seed {slot}")
            group[slot] = record
    cells = {}
    for (stage, layer, arm, split_name), group in grouped.items():
        ids = sorted({image for image, _ in group})
        seeds = sorted({seed for _, seed in group})
        if len(group) != len(ids) * len(seeds):
            raise ValueError(f"{path} {stage}_L{layer:02d}: not every image has every seed")
        values = {name: np.empty((len(seeds), len(ids))) for name in
                  ("predicted_count", "signed_error", "abs_error", "sq_error")}
        true = np.empty(len(ids))
        bins = []
        for i, image_id in enumerate(ids):
            truths = {group[(image_id, s)]["true_count"] for s in seeds}
            labels = {group[(image_id, s)]["bin"] for s in seeds}
            if len(truths) != 1 or len(labels) != 1:
                raise ValueError(f"{path} {image_id}: true count or bin differs across seeds")
            true[i] = float(truths.pop())
            label = labels.pop()
            if label != count_bin(true[i]):
                raise ValueError(f"{path} {image_id}: bin {label} for true count {true[i]:g}")
            bins.append(label)
            for s, seed in enumerate(seeds):
                record = group[(image_id, seed)]
                for name in values:
                    values[name][s, i] = float(record[name])
        signed = values["predicted_count"] - true[None, :]
        tolerance = 2e-4 + 2e-6 * np.abs(true)[None, :]
        if (np.abs(values["signed_error"] - signed) > tolerance).any():
            raise ValueError(f"{path} {stage}_L{layer:02d}: signed_error is not predicted minus true")
        if (np.abs(values["abs_error"] - np.abs(values["signed_error"])) > 2e-4).any():
            raise ValueError(f"{path} {stage}_L{layer:02d}: abs_error does not match signed_error")
        sq_tolerance = 2e-3 + 4e-4 * np.abs(values["signed_error"])
        if (np.abs(values["sq_error"] - values["signed_error"] ** 2) > sq_tolerance).any():
            raise ValueError(f"{path} {stage}_L{layer:02d}: sq_error does not match signed_error")
        tensor = {name: torch.from_numpy(value) for name, value in values.items()}
        cells[f"{stage}_L{layer:02d}"] = Cell(
            stage, layer, arm, split_name, ids, seeds, true, tensor["predicted_count"],
            tensor["signed_error"], tensor["abs_error"], tensor["sq_error"], bins,
        )
    return cells


# ------------------------------------------------------------------ result sources
@dataclass
class Result:
    """One model x arm x capacity result, from a complete payload or finished per-cell files."""

    dataset: str
    model: str
    arm: str
    capacity: str
    status: str                  # "complete" or "partial (finished cells only)"
    cells: dict[str, Cell]
    cell_json: dict[str, dict]   # key -> cell JSON
    sources: dict[str, str]      # path -> sha256
    val: dict[str, Cell] = field(default_factory=dict)

    def final(self, stage: str) -> Cell | None:
        """The deepest cell of a Stage (the final Tower for ``tower``)."""
        chosen = [c for c in self.cells.values() if c.stage == stage]
        return max(chosen, key=lambda c: c.layer) if chosen else None


def _check_against_json(cell: Cell, cell_json: dict, where: str, per_seed_key: str = "per_seed") -> None:
    """The TSV must reproduce the cell JSON's per-seed MAE (rounded to four places)."""
    if per_seed_key == "per_seed":
        recorded = [entry["mae"] for entry in cell_json.get("per_seed", [])]
    else:
        recorded = cell_json.get("metrics", {}).get("mae_by_seed", [])
    if not recorded:
        return
    mae = cell.abs_error.mean(dim=1).tolist()
    if len(recorded) != len(mae) or any(abs(a - b) > 2e-3 for a, b in zip(mae, recorded)):
        raise ValueError(f"{where}: TSV per-seed MAE {mae} does not match the cell JSON {recorded}")


def load_fsc(path: Path, allow_partial: bool = True) -> Result | None:
    """Load ``<model>_count_<arm>_<capacity>.json`` (or its ``.cells`` directory).

    A payload counts only with ``"complete": true`` and its predictions TSV. Without
    one, finished cells (a cell JSON with its predictions TSV) are used when
    ``allow_partial``; the result is then marked partial.
    """
    path = Path(path)
    stem = path.name.removesuffix(".json").removesuffix(".cells")
    match = RESULT_NAME.match(stem)
    if not match:
        raise ValueError(f"{path}: not a <model>_count_<arm>_<capacity> result")
    payload_path = path.with_name(f"{stem}.json")
    cells_dir = path.with_name(f"{stem}.cells")
    payload = json.loads(payload_path.read_text()) if payload_path.exists() else None
    sources: dict[str, str] = {}
    cell_json: dict[str, dict] = {}
    if payload is not None and payload.get("complete") is True:
        tsv = payload_path.with_suffix(".predictions.tsv")
        if not tsv.exists():
            raise FileNotFoundError(f"{tsv} is missing for a complete payload")
        cells = read_predictions(tsv, split="test")
        sources = {str(payload_path): sha256_file(payload_path), str(tsv): sha256_file(tsv)}
        for entry in payload["cells"]:
            cell_json[f"{entry['stage']}_L{entry['layer_index']:02d}"] = entry
        status = "complete"
        if set(cells) != set(cell_json):
            raise ValueError(f"{payload_path}: TSV cells {sorted(cells)} differ from JSON {sorted(cell_json)}")
    elif allow_partial and cells_dir.is_dir():
        cells = {}
        for cell_path in sorted(cells_dir.glob("*_L*.json")):
            if cell_path.name.endswith(".failed.json"):
                continue
            tsv = cell_path.with_suffix(".predictions.tsv")
            if not tsv.exists():
                continue
            found = read_predictions(tsv, split="test")
            if list(found) != [cell_path.stem]:
                raise ValueError(f"{tsv}: holds {sorted(found)}, expected {cell_path.stem}")
            cells |= found
            cell_json[cell_path.stem] = json.loads(cell_path.read_text())
            sources[str(cell_path)] = sha256_file(cell_path)
            sources[str(tsv)] = sha256_file(tsv)
        if not cells:
            return None
        status = "partial (finished cells only)"
    else:
        return None
    for key, cell in cells.items():
        cell.meta = {k: cell_json[key].get(k) for k in
                     ("model_id", "relative_depth", "learning_rate", "token_width", "input_width", "grid",
                      "trainable_parameters") if k in cell_json[key]}
        _check_against_json(cell, cell_json[key], f"{path} {key}")
    val = {}
    for key in cells:
        val_tsv = cells_dir / f"{key}.val.tsv"
        if val_tsv.exists():
            val |= read_predictions(val_tsv, split="val")
            sources[str(val_tsv)] = sha256_file(val_tsv)
    return Result("fsc147", match["model"], match["arm"], match["capacity"], status, cells, cell_json,
                  sources, val)


def load_countqa(path: Path) -> Result:
    """Load a ``scripts/countqa_eval.py`` payload and its predictions TSV (split ``countqa``)."""
    path = Path(path)
    match = RESULT_NAME.match(path.name.removesuffix(".json"))
    if not match:
        raise ValueError(f"{path}: not a <model>_count_<arm>_<variant> result")
    payload = json.loads(path.read_text())
    tsv = path.with_suffix(".predictions.tsv")
    cells = read_predictions(tsv, split="countqa")
    cell_json = {f"{c['stage']}_L{int(c['layer']):02d}": c for c in payload["cells"]}
    if set(cells) != set(cell_json):
        raise ValueError(f"{path}: TSV cells {sorted(cells)} differ from JSON {sorted(cell_json)}")
    for key, cell in cells.items():
        cell.meta = {"model_id": cell_json[key].get("model_id"),
                     "relative_depth": cell_json[key].get("relative_depth")}
        _check_against_json(cell, cell_json[key], f"{path} {key}", per_seed_key="metrics")
    return Result("countqa", match["model"], payload.get("arm", match["arm"]),
                  payload.get("variant", match["capacity"]), "complete", cells, cell_json,
                  {str(path): sha256_file(path), str(tsv): sha256_file(tsv)})


def discover(directory: Path, allow_partial: bool = True) -> list[Result]:
    """Every FSC result under a directory, complete payloads first, else finished cells."""
    directory = Path(directory)
    stems = sorted({p.name.removesuffix(".json").removesuffix(".cells") for p in directory.iterdir()
                    if RESULT_NAME.match(p.name.removesuffix(".json").removesuffix(".cells"))
                    and (p.suffix == ".json" or p.name.endswith(".cells"))}) if directory.is_dir() else []
    results = []
    for stem in stems:
        loaded = load_fsc(directory / f"{stem}.json", allow_partial)
        if loaded is not None:
            results.append(loaded)
    return results


# ------------------------------------------------------------------ statistics
def draws(images: int, resamples: int = RESAMPLES, seed: int = SEED, batch_size: int = BATCH) -> Iterator[torch.Tensor]:
    """Image index draws, identical to those of ``paired_metric_bootstrap`` for the same arguments."""
    generator = torch.Generator().manual_seed(seed)
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        yield torch.randint(images, (count, images), generator=generator)


def _interval(distribution: torch.Tensor, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    lower, upper = torch.quantile(distribution.double(), torch.tensor([tail, 1.0 - tail], dtype=torch.float64))
    return lower.item(), upper.item()


def seed_mean_rmse(sq_error: torch.Tensor) -> float:
    """Mean over seeds of each seed's RMSE; ``sq_error`` is (seeds, images)."""
    return sq_error.double().mean(dim=1).sqrt().mean().item()


def pooled_rmse(sq_error: torch.Tensor) -> float:
    """Square root of the mean squared error over all seed x image rows."""
    return math.sqrt(sq_error.double().mean().item())


def paired_rmse_bootstrap(
    first_sq: torch.Tensor,
    second_sq: torch.Tensor,
    *,
    resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = SEED,
    batch_size: int = BATCH,
) -> dict:
    """Paired image bootstrap of the RMSE difference, second minus first.

    ``first_sq`` and ``second_sq`` are (seeds, images) squared errors with seed k
    of one paired with seed k of the other. Each resample draws images, computes
    every seed's RMSE on the drawn images, and averages over seeds. The pooled
    RMSE difference uses the same draws.
    """
    if first_sq.shape != second_sq.shape or first_sq.ndim != 2:
        raise ValueError("squared errors must have the same (seeds, images) shape")
    if first_sq.shape[1] == 0:
        raise ValueError("cannot bootstrap an empty test split")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("resamples must be positive and confidence in (0, 1)")
    a, b = first_sq.double(), second_sq.double()
    seed_mean, pooled = [], []
    for index in draws(a.shape[1], resamples, seed, batch_size):
        da, db = a[:, index], b[:, index]          # (S, B, N)
        ma, mb = da.mean(dim=2), db.mean(dim=2)    # (S, B) per-seed MSE
        seed_mean.append(mb.sqrt().mean(dim=0) - ma.sqrt().mean(dim=0))
        pooled.append(mb.mean(dim=0).sqrt() - ma.mean(dim=0).sqrt())
    seed_mean, pooled = torch.cat(seed_mean), torch.cat(pooled)
    low, high = _interval(seed_mean, confidence)
    plow, phigh = _interval(pooled, confidence)
    return {
        "point_estimate": seed_mean_rmse(b) - seed_mean_rmse(a),
        "confidence": confidence, "lower": low, "upper": high,
        "resamples": resamples, "bootstrap_seed": seed, "difference": DIFFERENCE,
        "rmse": "mean over seeds of each seed's RMSE on the drawn images",
        "pooled": {"point_estimate": pooled_rmse(b) - pooled_rmse(a), "lower": plow, "upper": phigh,
                   "rmse": "sqrt of the mean squared error over all seed x drawn-image rows"},
    }


def cell_summary(cell: Cell) -> dict:
    """MAE, RMSE (seed-mean and pooled), bias and their per-seed values on the cell's images."""
    if not cell.image_ids:
        return {"images": 0}
    mae_by_seed = cell.abs_error.mean(dim=1)
    rmse_by_seed = cell.sq_error.mean(dim=1).sqrt()
    return {
        "images": len(cell.image_ids), "seeds": list(cell.seeds),
        "mae": mae_by_seed.mean().item(), "mae_std": mae_by_seed.std(unbiased=False).item(),
        "mae_by_seed": mae_by_seed.tolist(),
        "rmse": rmse_by_seed.mean().item(), "rmse_std": rmse_by_seed.std(unbiased=False).item(),
        "rmse_by_seed": rmse_by_seed.tolist(),
        "rmse_pooled": pooled_rmse(cell.sq_error),
        "bias": cell.signed.mean().item(),
        "mean_true_count": float(cell.true.mean()),
    }


def by_bin_summary(cell: Cell) -> dict:
    out = {}
    for name in BINS:
        ids = [i for i, b in zip(cell.image_ids, cell.bins) if b == name]
        out[name] = cell_summary(cell.subset(ids)) | {"small_sample": len(ids) < SMALL_BIN}
    return out


def _pair_check(first: Cell, second: Cell) -> None:
    if first.image_ids != second.image_ids:
        raise ValueError(f"{first.key} and {second.key} do not hold identical image IDs")
    if first.seeds != second.seeds:
        raise ValueError(f"{first.key} and {second.key} do not share seeds: {first.seeds} != {second.seeds}")
    if not np.array_equal(first.true, second.true):
        raise ValueError(f"{first.key} and {second.key} disagree on true counts")


def compare(first: Cell, second: Cell, *, resamples: int = RESAMPLES, confidence: float = CONFIDENCE,
            seed: int = SEED) -> dict:
    """Paired MAE and RMSE intervals, overall and per bin, on identical IDs and seeds."""
    _pair_check(first, second)

    def intervals(a: Cell, b: Cell) -> dict:
        mae = paired_metric_bootstrap(a.abs_error, b.abs_error, higher_is_better=False,
                                      resamples=resamples, confidence=confidence, seed=seed)
        rmse = paired_rmse_bootstrap(a.sq_error, b.sq_error, resamples=resamples, confidence=confidence,
                                     seed=seed)
        return {"mae": mae, "rmse": rmse}

    out = {
        "images": len(first.image_ids), "seeds_paired": list(first.seeds),
        "first_summary": cell_summary(first), "second_summary": cell_summary(second),
    } | intervals(first, second)
    bins = {}
    for name in BINS:
        ids = [i for i, b in zip(first.image_ids, first.bins) if b == name]
        entry: dict = {"images": len(ids), "small_sample": len(ids) < SMALL_BIN}
        if ids:
            a, b = first.subset(ids), second.subset(ids)
            entry |= {"first_summary": cell_summary(a), "second_summary": cell_summary(b)} | intervals(a, b)
        bins[name] = entry
    out["by_bin"] = bins
    return out


def resolved(interval: dict) -> str:
    """'first' or 'second' if the interval excludes zero in that cell's favour, else 'unresolved'."""
    if interval["lower"] > 0:
        return "first"
    if interval["upper"] < 0:
        return "second"
    return "unresolved"


# ------------------------------------------------------------------ comparison families
def describe(result: Result, cell: Cell) -> dict:
    return {"dataset": result.dataset, "model": result.model, "arm": result.arm, "capacity": result.capacity,
            "stage": cell.stage, "layer_index": cell.layer, "cell": cell.key, "status": result.status,
            **cell.meta}


def comparison(first: tuple[Result, Cell], second: tuple[Result, Cell], family: str, ids_rule: str,
               dropped: int = 0, **kwargs) -> dict:
    (ra, a), (rb, b) = first, second
    out = {
        "family": family,
        "first": describe(ra, a), "second": describe(rb, b),
        "difference": DIFFERENCE, "orientation": ORIENTATION,
        "image_ids_rule": ids_rule, "image_ids_sha256": sha256_ids(a.image_ids),
        "dropped_from_first": dropped,
    } | compare(a, b, **kwargs)
    out["image_ids"] = a.image_ids
    return out


def stage_comparisons(result: Result, **kwargs) -> dict[str, dict]:
    """Final Tower vs merged, merged vs projected, final Tower vs projected within one result."""
    out = {}
    for first, second in STAGE_PAIRS:
        a, b = result.final(first), result.final(second)
        if a is None or b is None:
            continue
        name = f"stage:{result.arm}_{result.capacity}:{first}_vs_{second}"
        out[name] = comparison((result, a), (result, b), "stage", "identical test IDs required", **kwargs)
    return out


def arm_comparison(free: Result, exemplar: Result, stage: str = "tower", **kwargs) -> dict[str, dict]:
    """Exemplar-free vs exemplar-conditioned at the final Tower on the exemplar arm's test IDs."""
    if free.capacity != exemplar.capacity or free.arm != "free" or exemplar.arm != "exemplar":
        raise ValueError("arm comparison needs a free and an exemplar result at the same capacity")
    a, b = free.final(stage), exemplar.final(stage)
    if a is None or b is None:
        return {}
    common = b.image_ids
    extra = sorted(set(common) - set(a.image_ids))
    if extra:
        raise ValueError(f"{len(extra)} exemplar-arm test IDs are not in the free arm, e.g. {extra[:3]}")
    name = f"arm:{stage}_{free.capacity}:free_vs_exemplar"
    return {name: comparison((free, a.subset(common)), (exemplar, b), "arm",
                             "common eligible test IDs: the exemplar arm's test IDs (intersection)",
                             dropped=len(a.image_ids) - len(common), **kwargs)}


def capacity_comparisons(raw: Result, matched: Result, **kwargs) -> dict[str, dict]:
    """Raw vs capacity-matched at each Stage present in both, same arm."""
    if raw.arm != matched.arm or raw.capacity != "raw" or matched.capacity != "matched":
        raise ValueError("capacity comparison needs raw and matched results of one arm")
    out = {}
    for stage in STAGES:
        a, b = raw.final(stage), matched.final(stage)
        if a is None or b is None:
            continue
        name = f"capacity:{raw.arm}_{stage}:raw_vs_matched"
        out[name] = comparison((raw, a), (matched, b), "capacity", "identical test IDs required", **kwargs)
    return out


def protocol(resamples: int = RESAMPLES, confidence: float = CONFIDENCE, seed: int = SEED) -> dict:
    return {
        "unit": "test image; seeds are never extra images",
        "pairing": "identical image IDs (sorted) and identical seeds; seed k of one cell with seed k of the other",
        "mae": "per-image absolute error averaged over seeds, then the paired difference of those means is "
               "resampled (vtb.geometry_bootstrap.paired_metric_bootstrap, higher_is_better=False)",
        "rmse": "per resample, each seed's RMSE on the drawn images, averaged over seeds, then differenced; "
                "'pooled' gives sqrt(mean sq_error over all seed x image rows) from the same draws",
        "draws": "one image draw sequence (torch.Generator seed, batches of 256) shared by MAE and RMSE",
        "point_estimate": "the same statistic on the full paired sample",
        "difference": ORIENTATION,
        "interval": f"{confidence:.0%} percentile",
        "resamples": resamples, "bootstrap_seed": seed, "confidence": confidence,
        "by_bin": f"resampled within the bin (counting.count_bin of the true count); small_sample when "
                  f"fewer than {SMALL_BIN} images",
        "stage_pairs": "final Tower vs merged, merged vs projected, final Tower vs projected, one arm and capacity",
        "arm": "free vs exemplar at the final Tower on the exemplar arm's test IDs, same capacity",
        "capacity": "raw vs matched at the same Stage and arm",
        "inputs": "contract prediction TSVs (abs_error, sq_error columns); complete payloads only, or finished "
                  "per-cell JSON + TSV pairs (marked partial)",
    }


# ------------------------------------------------------------------ qualitative selection
def select_dense(results: list[Result], k: int = 4, bin_name: str = "gt200") -> dict:
    """Deterministic choice of dense FSC test images whose error jumps across Stages.

    For every free-arm result holding all three final Stages, an image's jump is
    the largest absolute change in its seed-mean absolute error between any two
    Stages, divided by its true count. Results are pooled by taking, per image,
    the median jump across the qualifying results. The ``k`` largest medians
    win; ties break on the image ID.
    """
    jumps: dict[str, list[float]] = {}
    used, true = [], {}
    for result in sorted(results, key=lambda r: (r.model, r.capacity)):
        if result.dataset != "fsc147" or result.arm != "free":
            continue
        cells = [result.final(stage) for stage in STAGES]
        if any(c is None for c in cells):
            continue
        ids = cells[0].image_ids
        for c in cells[1:]:
            _pair_check(cells[0], c)
        used.append(f"{result.model}_count_{result.arm}_{result.capacity}")
        mean = torch.stack([c.abs_error.mean(dim=0) for c in cells])   # (3, N)
        spread = (mean.max(dim=0).values - mean.min(dim=0).values).numpy()
        for i, image_id in enumerate(ids):
            if cells[0].bins[i] != bin_name:
                continue
            true[image_id] = float(cells[0].true[i])
            jumps.setdefault(image_id, []).append(float(spread[i]) / max(true[image_id], 1.0))
    scored = sorted(((float(np.median(v)), image_id) for image_id, v in jumps.items()),
                    key=lambda item: (-item[0], item[1]))
    return {
        "rule": (f"FSC-147 test images in the {bin_name} bin. For each exemplar-free result with all three "
                 "final Stages (tower, merged, projected), an image's jump is max minus min over Stages of its "
                 "seed-mean absolute error, divided by its true count. Per image, take the median jump over "
                 f"those results; keep the {k} largest, ties broken by image ID ascending."),
        "bin": bin_name, "k": k, "results_used": used,
        "status": "selected" if scored else "pending: no free-arm result with all three Stages",
        "image_ids": [image_id for _, image_id in scored[:k]],
        "scores": [{"image_id": image_id, "median_relative_jump": score, "true_count": true[image_id],
                    "results": len(jumps[image_id])} for score, image_id in scored[:k]],
        "panel": "density maps only; FSC-147 pixels are not redistributed",
    }
