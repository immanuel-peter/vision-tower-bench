"""Train dense readouts on a cached run and report one geometry column."""

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn.functional import interpolate

from vtb import cache, counting, geometry, probe
from vtb.count_metrics import bins_for, density_loss, evaluate_count, summarise as summarise_counts
from vtb.geometry_metrics import (
    depth_si_loss,
    evaluate_depth,
    evaluate_surface_normal,
    normal_loss,
)
from vtb.probe_run import Split, fit_reducer, split_indices

TASKS = ("depth", "normal", "count")

# Validation metric and direction for each task. Count selects on validation MAE.
SELECTION = {"depth": ("d1", True), "normal": ("mean_deg", False), "count": ("abs_error", False)}

# Six points spanning both edges the four-point matrix truncated at.
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
# Count heads explode in the first epoch at 1e-2 and above and select at the old
# floor, so the same six-point budget sits one decade lower (count-v3).
COUNT_LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)


def manifest(targets: Path) -> dict:
    path = targets.with_name(targets.name.replace("_targets.npz", "_manifest.json"))
    return json.loads(path.read_text()) if path.exists() else {}


def depth_range(targets: Path, override: float | None) -> float:
    if override:
        return override
    return float(manifest(targets).get("max_depth_metres") or 10.0)


def selected_slices(
    run: Path,
    only: list[str] | None = None,
    stages: list[str] | None = None,
    deepest_only: bool = False,
) -> list[tuple[str, int]]:
    chosen = cache.slices(run)
    if only:
        wanted = {(value.split(":")[0], int(value.split(":")[1])) for value in only}
        chosen = [cell for cell in chosen if cell in wanted]
        if len(chosen) != len(wanted):
            raise ValueError(f"{only} does not match slices in {run}")
    if stages:
        wanted_stages = set(stages)
        chosen = [cell for cell in chosen if cell[0] in wanted_stages]
    if deepest_only:
        deepest = max(layer for stage, layer in cache.slices(run) if stage == "tower")
        chosen = [cell for cell in chosen if cell[1] == deepest]
    return chosen


def centre_square(target):
    height, width = target.shape[-2:]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return target[..., top:top + side, left:left + side]


def load_targets(path: Path, image_ids: list[str], task: str):
    store = np.load(path)

    def stack(suffix: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([centre_square(store[f"{i}{suffix}"]) for i in image_ids])).float()

    if task == "normal":
        return stack("_normal"), stack("_valid").unsqueeze(1)
    return stack("").unsqueeze(1), None


def coverage_of(targets: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    mask = valid if valid is not None else (targets > 0).float()
    return mask.flatten(1).mean(dim=1)


class Diverged(RuntimeError):
    """A readout's training loss became non-finite."""


def train_cell(features, targets, valid, split, task, args, learning_rate, seed, monitor=None):
    """Fit one head. ``monitor(epoch, model, mean_loss)`` runs after each epoch when given."""
    torch.manual_seed(seed)
    model = build_head(features.shape[1], task, args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        order = split.train[torch.randperm(len(split.train))]
        total = torch.zeros((), device=args.device)
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            loss = _loss(model, features, targets, valid, index, task, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if monitor is not None:
                total += loss.detach() * len(index)
        schedule.step()
        if monitor is not None:
            mean_loss = total.item() / len(order)
            if not math.isfinite(mean_loss):
                raise Diverged(f"loss {mean_loss} at epoch {epoch + 1}, lr {learning_rate:g}, seed {seed}")
            monitor(epoch + 1, model, mean_loss)

    return model


def _predict(model, features, index, device, target_hw):
    prediction = model([features[index].to(device).float()])
    # CountHead already predicts on the declared target grid.
    if tuple(prediction.shape[-2:]) == tuple(target_hw):
        return prediction
    return interpolate(prediction, target_hw, mode="bilinear", align_corners=False)


def _loss(model, features, targets, valid, index, task, device):
    batch = targets[index].to(device)
    prediction = _predict(model, features, index, device, batch.shape[-2:])
    if task == "depth":
        return depth_si_loss(prediction, batch)
    if task == "count":
        return density_loss(prediction, batch)
    return normal_loss(prediction, batch, valid[index].to(device))


@torch.inference_mode()
def score(model, features, targets, valid, index, task, args) -> dict[str, torch.Tensor]:
    """Per-image metrics. For count, ``valid`` holds each image's integer retained count."""
    model.eval()
    totals: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(index), args.batch_size):
        chunk = index[start:start + args.batch_size]
        batch = targets[chunk].to(args.device)
        prediction = _predict(model, features, chunk, args.device, batch.shape[-2:])
        if task == "depth":
            metrics = evaluate_depth(prediction, batch, scale_invariant=args.scale_invariant)
        elif task == "count":
            metrics = evaluate_count(prediction, valid[chunk])
        else:
            metrics = evaluate_surface_normal(prediction, batch, valid[chunk].to(args.device))
        for key, value in metrics.items():
            totals.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(v) for key, v in totals.items()}


def summarise(metrics, coverage, image_ids, index, scenes) -> dict:
    rows: dict[str, list[int]] = {}
    for position, image_id in enumerate(index.tolist()):
        rows.setdefault(scenes.get(image_ids[image_id], "unknown"), []).append(position)
    coverage = coverage[index]

    def block(chosen: torch.Tensor) -> dict:
        out = {key: round(value[chosen].mean().item(), 4) for key, value in metrics.items()}
        return out | {"images": len(chosen), "coverage": round(coverage[chosen].mean().item(), 4)}

    everything = block(torch.arange(len(index)))
    return everything | {
        "by_scene": {name: block(torch.tensor(sorted(r))) for name, r in sorted(rows.items())}
    }


def select_learning_rate(features, targets, valid, split, task, args, curves=None) -> tuple[float, dict]:
    """Search rates on validation. With ``curves``, record per-epoch curves and survive divergence.

    For count, a selection on the edge of the searched rates extends the search
    outward by half a decade, up to EDGE_EXTENSIONS times, stopping when the new
    rate collapses or does not improve.
    """
    metric, higher_is_better = SELECTION[task]
    searched: dict[str, float] = {}
    best_rate, best_score = None, None
    rates = list(args.learning_rates)
    extended: list[float] = []
    for rate in iter_rates(rates, extended, lambda: best_rate, task == "count" and curves is not None):
        if curves is None:
            model = train_cell(features, targets, valid, split, task, args, rate, seed=0)
            value = score(model, features, targets, valid, split.val, task, args)[metric].mean().item()
        else:
            curve = curves.setdefault(f"lr {rate:g}", {})
            began = time.perf_counter()
            try:
                model = train_cell(features, targets, valid, split, task, args, rate, seed=0,
                                   monitor=epoch_monitor(curve, features, targets, valid, split, task, args))
                metrics = score(model, features, targets, valid, split.val, task, args)
                if task == "count" and (reason := collapsed(metrics)):
                    raise Diverged(f"lr {rate:g}: {reason}")
                value = metrics[metric].mean().item()
            except Diverged as error:
                curve["diverged"] = str(error)
                value = math.nan
            curve["seconds"] = round(time.perf_counter() - began, 1)
        searched[f"{rate:g}"] = round(value, 4)
        if not math.isfinite(value):
            continue
        if best_score is None or (value > best_score if higher_is_better else value < best_score):
            best_rate, best_score = rate, value
    if best_rate is None:
        raise Diverged(f"every learning rate diverged: {searched}")
    selection = {
        "learning_rate": best_rate,
        "val_metric": metric,
        "val_score": round(best_score, 4),
        "learning_rate_search": searched,
    }
    if task == "count" and curves is not None:
        selection["learning_rate_extended"] = extended
        selection["learning_rate_at_edge"] = best_rate in (min(searched_rates(searched)), max(searched_rates(searched)))
    return best_rate, selection


EDGE_EXTENSIONS = 2


def searched_rates(searched: dict[str, float]) -> list[float]:
    return [float(rate) for rate in searched]


def half_decade(rate: float, up: bool) -> float:
    """The next point on a 1, 3, 10, 30 grid: 3e-3 -> 1e-2 up, 1e-5 -> 3e-6 down."""
    exponent = math.floor(math.log10(rate) + 1e-9)
    mantissa = round(rate / 10 ** exponent)
    if up:
        return 3 * 10.0 ** exponent if mantissa == 1 else 10.0 ** (exponent + 1)
    return 10.0 ** exponent if mantissa == 3 else 3 * 10.0 ** (exponent - 1)


def iter_rates(rates: list[float], extended: list[float], best, extend: bool):
    """Yield the grid, then up to EDGE_EXTENSIONS half-decade steps past whichever edge wins.

    Extension stops as soon as the newest rate does not win (it collapsed or scored worse).
    """
    tried = list(rates)
    yield from rates
    while extend and len(extended) < EDGE_EXTENSIONS:
        current = best()
        if current is None or (extended and current != extended[-1]):
            return
        if current == max(tried):
            step = half_decade(current, up=True)
        elif current == min(tried):
            step = half_decade(current, up=False)
        else:
            return
        extended.append(step)
        tried.append(step)
        yield step


def _mean_std(values: list[float]) -> tuple[float, float]:
    stacked = torch.tensor(values, dtype=torch.float64)
    return round(stacked.mean().item(), 4), round(stacked.std(unbiased=False).item(), 4)


def aggregate(runs: list[dict]) -> dict:
    out: dict = {}
    for key, value in runs[0].items():
        if key in ("by_scene", "by_bin"):
            out[key] = {name: aggregate([r[key][name] for r in runs]) for name in value}
        elif key == "exemplar_eligible_subset":
            out[key] = aggregate([r[key] for r in runs])
        elif key in ("images", "coverage"):
            out[key] = value
        else:
            out[key], out[f"{key}_std"] = _mean_std([r[key] for r in runs])
    return out


def build_head(width: int, task: str, args) -> torch.nn.Module:
    if task == "depth":
        return geometry.DepthHead([width], head=args.head, max_depth=args.max_depth)
    if task == "count":
        return geometry.CountHead([width], head=args.head, extent=getattr(args, "grid_extent", 1.0))
    return geometry.SurfaceNormalHead([width], head=args.head)


def match_capacity(batch, split, width: int):
    reducer = fit_reducer(batch.tokens.float(), split, width)
    return batch.with_tokens(reducer(batch.tokens.float()))


def run_cell(features, targets, valid, split, task, coverage, image_ids, scenes, args) -> dict:
    best_rate, selection = select_learning_rate(features, targets, valid, split, task, args)
    per_seed = []
    for seed in range(args.seeds):
        model = train_cell(features, targets, valid, split, task, args, best_rate, seed)
        metrics = score(model, features, targets, valid, split.test, task, args)
        per_seed.append(summarise(metrics, coverage, image_ids, split.test, scenes))
    return aggregate(per_seed) | selection | {"seeds": args.seeds, "per_seed": per_seed}


# Count readouts. FSC-147 uses its official class-disjoint split, and every final
# seed's head, PCA and per-image predictions are kept for transfer and bootstrap.

PREDICTION_COLUMNS = (
    "image_id", "split", "stage", "layer", "arm", "seed", "true_count",
    "predicted_count", "signed_error", "abs_error", "sq_error", "bin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    scratch = path.with_name(f".{path.name}.{os.getpid()}.partial")
    scratch.write_text(text)
    os.replace(scratch, path)


def official_split(image_ids: list[str], splits: dict[str, list[str]], keep: set[str] | None = None) -> Split:
    """Index the cache rows by the manifest's split lists. The cache must hold exactly those IDs."""
    position = {image_id: row for row, image_id in enumerate(image_ids)}
    listed = [image_id for name in ("train", "val", "test") for image_id in splits[name]]
    if len(position) != len(image_ids) or len(set(listed)) != len(listed):
        raise ValueError("duplicate image IDs in the cache or the split lists")
    if set(listed) != set(image_ids):
        raise ValueError(
            f"cache and split differ: {len(set(image_ids) - set(listed))} unlisted, "
            f"{len(set(listed) - set(image_ids))} missing"
        )

    def rows(name: str) -> torch.Tensor:
        chosen = [position[i] for i in splits[name] if keep is None or i in keep]
        return torch.tensor(chosen, dtype=torch.long)

    return Split(rows("train"), rows("val"), rows("test"))


def count_targets(path: Path, image_ids: list[str], images: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 1, G, G) density and (N,) integer retained counts, checked against each other."""
    density, _ = load_targets(path, image_ids, "count")
    counts = torch.tensor([images[i]["count"] for i in image_ids], dtype=torch.float64)
    mass = density.double().flatten(1).sum(dim=1)
    if not torch.allclose(mass, counts, atol=1e-3 + 1e-5 * counts.max().item()):
        worst = (mass - counts).abs().argmax().item()
        raise ValueError(f"{image_ids[worst]} density sums to {mass[worst]:.4f}, count {counts[worst]:.0f}")
    return density, counts


def exemplar_boxes(images: dict, image_ids: list[str], rows: list[int], limit: int | None = None) -> list[np.ndarray]:
    boxes = []
    for row in rows:
        entry = images[image_ids[row]]
        valid = [box for box, ok in zip(entry["boxes"], entry["box_valid"]) if ok]
        boxes.append(np.asarray(valid[:limit], dtype=np.float64).reshape(-1, 4))
    return boxes


def grid_extent(model_id: str, stage: str, resolution: int, side: int) -> float:
    """Canvas fraction the patch grid spans. Two adapters pad the image bottom-right first.

    Gemma 4 pads to a multiple of its 3x3 pool before patching; DeepSeek V4.1
    pads the Tower grid to a multiple of its 3x3 merge. Every other adapter's
    grid covers exactly the image.
    """
    from vtb.adapters import deepseek_v41, gemma4

    extent = 1.0
    if model_id == gemma4.MODEL_ID:
        extent = gemma4.padded_side(resolution) / resolution
        pixels = gemma4.PATCH_SIZE * (1 if stage == "tower" else gemma4.POOL_SIZE)
    elif model_id == deepseek_v41.MODEL_ID and stage != "tower":
        patches = resolution // deepseek_v41.PATCH_SIZE
        cells = -(-patches // deepseek_v41.DOWNSAMPLE)
        extent = cells * deepseek_v41.DOWNSAMPLE * deepseek_v41.PATCH_SIZE / resolution
        pixels = deepseek_v41.PATCH_SIZE * deepseek_v41.DOWNSAMPLE
    else:
        return extent
    if round(side * pixels) != round(extent * resolution):
        raise ValueError(f"{model_id} {stage}: grid {side} x {pixels} px does not span {extent * resolution:.0f} px")
    return extent


def exemplar_channel(features: torch.Tensor, boxes: list[np.ndarray], rows: list[int], extent: float = 1.0,
                     chunk: int = 256) -> torch.Tensor:
    """The conditioning channel for ``rows``; other rows stay zero and never enter a split."""
    channel = torch.zeros(features.shape[0], 1, *features.shape[-2:], dtype=features.dtype)
    for start in range(0, len(rows), chunk):
        index = rows[start:start + chunk]
        channel[index] = counting.prototype_similarity(
            features[index].to(features.device), boxes[start:start + chunk], extent=extent
        ).cpu()
    return channel


def epoch_monitor(curve: dict, features, targets, valid, split, task, args):
    metric = SELECTION[task][0]

    def record(epoch: int, model, mean_loss: float) -> None:
        value = score(model, features, targets, valid, split.val, task, args)[metric].mean().item()
        curve.setdefault("epoch", []).append(epoch)
        curve.setdefault("train_loss", []).append(round(mean_loss, 6))
        curve.setdefault(f"val_{metric}", []).append(round(value, 4))

    return record


def prediction_rows(metrics, image_ids, index, split_name, stage, layer, arm, seed) -> list[str]:
    rows = []
    bins = bins_for(metrics["true_count"].tolist())
    for position, row in enumerate(index.tolist()):
        values = (
            image_ids[row], split_name, stage, layer, arm, seed,
            int(round(metrics["true_count"][position].item())),
            f"{metrics['predicted_count'][position].item():.4f}",
            f"{metrics['signed_error'][position].item():.4f}",
            f"{metrics['abs_error'][position].item():.4f}",
            f"{metrics['sq_error'][position].item():.4f}",
            bins[position],
        )
        rows.append("\t".join(str(v) for v in values))
    return rows


def save_readout(directory: Path, seed: int, model: torch.nn.Module, reducer, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if reducer is not None:
        pca = directory / "pca.safetensors"
        scratch = pca.with_name(f".{pca.name}.{os.getpid()}.partial")
        save_file({"mean": reducer.mean.contiguous(), "basis": reducer.basis.contiguous()}, str(scratch),
                  metadata={"meta": json.dumps({"fit_split": "train", "width": reducer.basis.shape[1],
                                                "identity": meta["identity"]})})
        os.replace(scratch, pca)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    path = directory / f"seed{seed}.safetensors"
    scratch = path.with_name(f".{path.name}.{os.getpid()}.partial")
    save_file(state, str(scratch), metadata={"meta": json.dumps(meta | {"seed": seed, "pca": reducer is not None})})
    os.replace(scratch, path)


@dataclass
class Readout:
    """A saved count head with its PCA, ready for frozen transfer."""

    head: torch.nn.Module
    reducer: probe.Reducer | None
    conditioned: bool
    meta: dict

    @torch.inference_mode()
    def density(self, tokens: torch.Tensor, boxes: list[np.ndarray] | None = None,
                valid_rects: list[tuple[float, float, float, float]] | None = None) -> torch.Tensor:
        """(N, T, D) cache tokens to (N, 1, G, G) density, on the training numerical path.

        ``boxes`` are unit canvas boxes (conditioned readouts only); ``valid_rects`` are
        each image's valid canvas rectangle, e.g. a letterbox frame's ``valid``.
        """
        features = tokens.float().cpu()
        if self.reducer is not None:
            features = self.reducer(features)
        rows, count, width = features.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError(f"{count} tokens do not form a square grid")
        features = features.transpose(1, 2).reshape(rows, width, side, side).contiguous()
        if self.conditioned:
            if boxes is None:
                raise ValueError("this readout needs exemplar boxes")
            extent = self.meta["grid_extent"]
            valid = None
            if valid_rects is not None:
                valid = torch.from_numpy(np.stack([counting.token_valid(r, side, extent) for r in valid_rects]))
            features = counting.condition(features, boxes, valid, extent)
        elif boxes is not None:
            raise ValueError("this readout is exemplar-free")
        device = next(self.head.parameters()).device
        return self.head([features.to(device)])


def load_readout(directory: Path, seed: int, device: str = "cuda") -> Readout:
    path = Path(directory) / f"seed{seed}.safetensors"
    with safe_open(path, framework="pt") as handle:
        meta = json.loads(handle.metadata()["meta"])
    head = geometry.CountHead(
        [meta["input_dim"]], head=meta["head"], hidden_dim=meta["hidden_dim"],
        grid=meta["grid"], scale=meta["density_scale"], extent=meta["grid_extent"],
    )
    head.load_state_dict(load_file(path))
    reducer = None
    if meta["pca"]:
        saved = load_file(Path(directory) / "pca.safetensors")
        reducer = probe.Reducer(basis=saved["basis"], mean=saved["mean"])
    return Readout(head.to(device).eval(), reducer, meta["arm"] == "exemplar", meta)


def count_identity(args, stage: str, layer: int, provenance: dict) -> str:
    """Hash of everything that decides a count cell's numbers, for safe resumption."""
    fields = {
        "protocol": counting.PROTOCOL, "grid": counting.TARGET_GRID, "sigma": counting.SIGMA_CELLS,
        "truncate": counting.TRUNCATE_SIGMAS, "scale": counting.DENSITY_SCALE,
        "arm": args.arm, "matched": args.match_capacity, "extent": getattr(args, "grid_extent", 1.0),
        "edge_extensions": EDGE_EXTENSIONS, "width": args.width, "head": args.head,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rates": [f"{r:g}" for r in args.learning_rates], "seeds": args.seeds,
        "stage": stage, "layer": layer,
    } | provenance
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def readout_identity(path: Path) -> str | None:
    if not path.exists():
        return None
    with safe_open(path, framework="pt") as handle:
        return json.loads(handle.metadata()["meta"]).get("identity")


def finished_cell(cells_dir: Path, key: str, identity: str, readouts: Path | None, seeds: int,
                  matched: bool = False) -> dict | None:
    """A finished cell whose result, predictions and every saved readout carry this identity."""
    path = cells_dir / f"{key}.json"
    if not path.exists():
        return None
    cell = json.loads(path.read_text())
    if cell.get("identity") != identity or not (cells_dir / f"{key}.predictions.tsv").exists():
        return None
    if readouts is not None:
        files = [readouts / f"seed{s}.safetensors" for s in range(seeds)]
        if matched:
            files.append(readouts / "pca.safetensors")
        if any(readout_identity(f) != identity for f in files):
            return None
    return cell


def collapsed(metrics: dict[str, torch.Tensor]) -> str | None:
    """Name a degenerate count map: near-zero total mass, or the same count for every image."""
    predicted, true = metrics["predicted_count"], metrics["true_count"]
    ratio = predicted.sum().item() / max(true.sum().item(), 1e-9)
    if ratio < COLLAPSE_MASS:
        return f"zero map: predicted mass is {ratio:.4f} of the true mass"
    if predicted.std().item() < 1e-3 * max(predicted.mean().item(), 1e-9):
        return f"constant map: every image predicts {predicted.mean().item():.3f}"
    return None


# A fit predicting under a tenth of the validation mass has collapsed.
COLLAPSE_MASS = 0.1


# A collapsed final seed retrains every seed at the next-best validation rate, at most this often.
SEED_FALLBACKS = 2


def run_count_cell(features, targets, counts, split, image_ids, images, stage, layer, reducer,
                   readouts, meta, args) -> tuple[dict, list[str], list[str]]:
    curves: dict = {}
    best_rate, selection = select_learning_rate(features, targets, counts, split, "count", args, curves.setdefault("search", {}))
    # Rates by validation MAE. A final seed collapsing at the selected rate marks it unstable;
    # all seeds then retrain at the next rate, so a published cell never averages a collapse.
    ranked = sorted((value, float(rate)) for rate, value in selection["learning_rate_search"].items()
                    if math.isfinite(value))
    candidates = [best_rate] + [rate for _, rate in ranked if rate != best_rate][:SEED_FALLBACKS]
    fallbacks = []
    for rate in candidates:
        try:
            outcome = train_seeds(features, targets, counts, split, image_ids, images, stage, layer, rate, curves, args)
            break
        except Diverged as error:
            fallbacks.append({"learning_rate": rate, "reason": str(error)})
    else:
        raise Diverged(f"final seeds collapsed at every candidate rate: {fallbacks}")
    per_seed, one_exemplar, test_rows, val_rows, models = outcome
    if readouts is not None:
        for seed, model in enumerate(models):
            save_readout(readouts, seed, model, reducer, meta | {"learning_rate": rate})
    # RMSE: each seed's from its squared errors, then mean and std over seeds as for every
    # metric; rmse_pooled is sqrt of the mean squared error over all seed-image rows.
    squared = [error for summary in per_seed for error in summary.pop("rmse_pooled_input")]
    cell = aggregate(per_seed) | selection | {"seeds": args.seeds, "per_seed": per_seed, "curves": curves}
    cell["rmse_pooled"] = round(math.sqrt(sum(squared) / len(squared)), 4)
    if fallbacks:
        cell |= {"learning_rate_selected": best_rate, "learning_rate": rate, "seed_fallbacks": fallbacks}
    if one_exemplar:
        cell["one_exemplar_val"] = {"note": "transfer diagnostic only; never used for selection",
                                    "per_seed": one_exemplar} | aggregate(one_exemplar)
    return cell, test_rows, val_rows


def train_seeds(features, targets, counts, split, image_ids, images, stage, layer, rate, curves, args):
    """Fit and score every final seed at one rate. Raises Diverged when any seed collapses."""
    per_seed, one_exemplar, test_rows, val_rows, models = [], [], [], [], []
    for seed in range(args.seeds):
        curve = curves.setdefault(f"seed {seed}", {})
        curve.clear()
        curve["learning_rate"] = rate
        began = time.perf_counter()
        model = train_cell(features, targets, counts, split, "count", args, rate, seed,
                           monitor=epoch_monitor(curve, features, targets, counts, split, "count", args))
        test = score(model, features, targets, counts, split.test, "count", args)
        val = score(model, features, targets, counts, split.val, "count", args)
        curve["seconds"] = round(time.perf_counter() - began, 1)
        if reason := collapsed(val):
            curves.setdefault("collapsed_final_seeds", []).append(
                {"learning_rate": rate, "seed": seed, "reason": reason, "curve": dict(curve)})
            raise Diverged(f"final seed {seed} at lr {rate:g}: {reason}")
        summary = summarise_counts(test, bins_for(test["true_count"].tolist()))
        if args.arm == "free":
            # The exemplar arm's test images, for arm comparisons on common IDs.
            common = [k for k, row in enumerate(split.test.tolist()) if images[image_ids[row]]["exemplar_eligible"]]
            subset = {key: value[common] for key, value in test.items()}
            summary["exemplar_eligible_subset"] = summarise_counts(subset, bins_for(subset["true_count"].tolist()))
        summary["rmse_pooled_input"] = test["sq_error"].tolist()
        per_seed.append(summary)
        test_rows += prediction_rows(test, image_ids, split.test, "test", stage, layer, args.arm, seed)
        val_rows += prediction_rows(val, image_ids, split.val, "val", stage, layer, args.arm, seed)
        if args.arm == "exemplar":
            one_exemplar.append(one_exemplar_val(model, features, targets, counts, split, image_ids, images, args))
        models.append(model.cpu())
    return per_seed, one_exemplar, test_rows, val_rows, models


def one_exemplar_val(model, features, targets, counts, split, image_ids, images, args) -> dict:
    """Validation with only the first release box, read by the same three-box head."""
    rows = split.val.tolist()
    saved = features[split.val, -1:].clone()
    single = exemplar_channel(features[split.val, :-1], exemplar_boxes(images, image_ids, rows, limit=1),
                              list(range(len(rows))), args.grid_extent)
    features[split.val, -1:] = single.to(features.device)
    try:
        metrics = score(model, features, targets, counts, split.val, "count", args)
    finally:
        features[split.val, -1:] = saved
    return summarise_counts(metrics, bins_for(metrics["true_count"].tolist()))


def count_main(args) -> None:
    targets_manifest = manifest(args.targets)
    images, splits = targets_manifest["images"], targets_manifest["splits"]
    if targets_manifest.get("protocol") != counting.PROTOCOL:
        raise SystemExit(f"targets were prepared for {targets_manifest.get('protocol')}, not {counting.PROTOCOL}")
    completion = cache.completion(args.run)
    # The features must come from exactly the prepared images these targets describe.
    if completion.get("png_digest") != counting.png_digest(images):
        raise SystemExit(f"{args.run} was extracted from other images than {args.targets}")
    provenance = {"cache": completion["sha256"], "targets": sha256_file(args.targets),
                  "manifest": sha256_file(args.targets.with_name(args.targets.name.replace("_targets.npz", "_manifest.json")))}
    capacity = "matched" if args.match_capacity else "raw"
    name = args.name or args.run.name
    out = args.out or args.run / f"{name}_count_{args.arm}_{capacity}.json"
    cells_dir = out.with_suffix(".cells")
    cells_dir.mkdir(parents=True, exist_ok=True)

    try:
        chosen = selected_slices(args.run, args.only, args.stages, args.deepest_only)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.arm == "exemplar" and any(stage != "tower" for stage, _ in chosen):
        raise SystemExit("the exemplar arm is Tower-only; pass --stages tower --deepest-only")

    cells = []
    for stage, layer in chosen:
        key = f"{stage}_L{layer:02d}"
        recorded = completion["slices"][key]
        args.grid_extent = grid_extent(recorded["model_id"], stage, int(recorded["resolution"]),
                                       math.isqrt(recorded["tokens"]))
        identity = count_identity(args, stage, layer, provenance)
        readouts = args.readouts / name / f"{args.arm}_{capacity}" / key if args.readouts else None
        if done := finished_cell(cells_dir, key, identity, readouts, args.seeds, args.match_capacity):
            print(f"{stage:>9} L{layer:02d}  already complete, skipping", flush=True)
            cells.append(done)
            continue

        started = clock = time.perf_counter()
        timing: dict[str, float] = {}

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timing[name] = round(now - clock, 2)
            clock = now

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        batch = cache.load_batch(args.run, stage, layer)
        order, token_width = batch.image_ids, batch.tokens.shape[-1]
        eligible = {i for i in order if images[i]["exemplar_eligible"]} if args.arm == "exemplar" else None
        split = official_split(order, splits, eligible)
        targets, counts = count_targets(args.targets, order, images)
        lap("load")

        reducer = None
        if args.match_capacity:
            # PCA always fits on the whole official training split, so both arms share it.
            # Reducer.fit upcasts to float32 itself, so the bfloat16 cache gives the same fit.
            reducer = fit_reducer(batch.tokens, official_split(order, splits), args.width)
            lap("pca_fit")
            batch = batch.with_tokens(torch.cat([
                reducer(batch.tokens[start:start + 512].float())
                for start in range(0, len(order), 512)
            ]))
        # bfloat16 to float32 is exact, so converting after the move changes no value.
            lap("pca_transform")
        features = geometry.dense_map(batch).to(args.features_device).float()
        if args.arm == "exemplar":
            rows = sorted(torch.cat([split.train, split.val, split.test]).tolist())
            channel = exemplar_channel(features, exemplar_boxes(images, order, rows), rows, args.grid_extent)
            features = torch.cat([features, channel.to(features.device)], dim=1)
        lap("prepare_features")

        parameters = geometry.parameter_count(build_head(features.shape[1], "count", args))
        meta = {
            "protocol": counting.PROTOCOL, "task": "count", "arm": args.arm, "capacity": capacity,
            "input_dim": features.shape[1], "head": args.head,
            "hidden_dim": build_head(features.shape[1], "count", args).head.conv_mid[0].out_channels,
            "identity": identity, "resolution": batch.resolution, "tokens": features.shape[-1] ** 2,
            "code": code_commit(),
            "grid": counting.TARGET_GRID, "density_scale": counting.DENSITY_SCALE,
            "grid_extent": args.grid_extent, "epochs": args.epochs, "batch_size": args.batch_size, "model_id": batch.model_id,
            "stage": stage, "layer": layer, "feature_cache": str(args.run),
            "feature_cache_sha256": provenance["cache"], "targets_sha256": provenance["targets"],
            "manifest_sha256": provenance["manifest"], "torch": torch.__version__,
            "precision": precision(),
        }
        try:
            cell, test_rows, val_rows = run_count_cell(
                features, targets, counts, split, order, images, stage, layer, reducer, readouts, meta, args
            )
        except Diverged as error:
            write_atomic(cells_dir / f"{key}.failed.json", json.dumps({"error": str(error), "identity": identity}) + "\n")
            print(f"{stage:>9} L{layer:02d}  FAILED {error}", flush=True)
            continue
        lap("train_and_score")
        cell |= {
            "protocol": counting.PROTOCOL, "identity": identity, "model_id": batch.model_id, "stage": stage, "layer_index": layer,
            "relative_depth": batch.relative_depth, "token_width": token_width,
            "input_width": features.shape[1], "grid": features.shape[-1],
            "trainable_parameters": parameters,
            "images": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
            "readout_dir": str(readouts) if readouts else None,
            "seconds": round(time.perf_counter() - started, 1),
            "timing_seconds": timing,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
            "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        }
        header = "\t".join(PREDICTION_COLUMNS) + "\n"
        write_atomic(cells_dir / f"{key}.val.tsv", header + "\n".join(val_rows) + "\n")
        write_atomic(cells_dir / f"{key}.predictions.tsv", header + "\n".join(test_rows) + "\n")
        write_atomic(cells_dir / f"{key}.json", json.dumps(cell, indent=2) + "\n")
        cells.append(cell)
        print(
            f"{stage:>9} L{layer:02d}  MAE {cell['mae']:.2f} +- {cell['mae_std']:.2f}  "
            f"RMSE {cell['rmse']:.2f}  lr {cell['learning_rate']:g}  params {parameters:,}  "
            f"{cell['seconds']:.0f}s",
            flush=True,
        )
        del features, batch

    expected = len(chosen)
    payload = {
        "task": "count", "protocol": counting.PROTOCOL, "dataset": targets_manifest.get("dataset", "FSC-147"),
        "arm": args.arm, "capacity_matched": args.match_capacity, "head": args.head,
        "target_grid": counting.TARGET_GRID, "sigma_cells": counting.SIGMA_CELLS,
        "density_scale": counting.DENSITY_SCALE, "epochs": args.epochs, "batch_size": args.batch_size,
        "seeds": args.seeds, "learning_rates": args.learning_rates, "split": "official",
        "expected_cells": expected, "complete": len(cells) == expected,
        "provenance": provenance, "cells": cells,
    }
    write_atomic(out, json.dumps(payload, indent=2) + "\n")
    header = "\t".join(PREDICTION_COLUMNS) + "\n"
    rows = [line for cell in cells for line in
            (cells_dir / f"{cell['stage']}_L{cell['layer_index']:02d}.predictions.tsv").read_text().splitlines()[1:]]
    write_atomic(out.with_suffix(".predictions.tsv"), header + "\n".join(rows) + "\n")
    print(f"\nwrote {out} ({len(cells)}/{expected} cells)")
    if len(cells) != expected:
        raise SystemExit(f"{expected - len(cells)} cells failed; see {cells_dir}")


def code_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return found.stdout.strip() or "unknown"


def check_readout(readout: Readout, batch) -> None:
    """Refuse a readout trained on another model, Stage, layer, resolution or grid than the cache."""
    expected = {"model_id": batch.model_id, "stage": batch.stage, "layer": batch.layer_index,
                "resolution": batch.resolution, "tokens": batch.tokens.shape[1]}
    found = {key: readout.meta.get(key) for key in expected}
    if found != expected:
        raise ValueError(f"readout {found} does not match cache {expected}")


def precision() -> dict:
    return {
        "features": "bfloat16 cache, float32 readout",
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the dense readouts on a cached run.")
    ap.add_argument("--run", type=Path, required=True, help="full-token cache from vtb.extract")
    ap.add_argument("--targets", type=Path, required=True, help="npz written by the dataset prep step")
    ap.add_argument("--task", default="depth", choices=TASKS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--head", default="multiscale", choices=("multiscale", "linear"))
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument(
        "--match-capacity",
        action="store_true",
        help="reduce each Stage to --width before training the head",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="search these rates on validation for every cell (default: the task's grid)",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scale-invariant", action="store_true")
    ap.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="depth ceiling in metres; defaults to the prep manifest",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", default="free", choices=("free", "exemplar"),
                    help="count only: exemplar-free, or conditioned on the release exemplar boxes")
    ap.add_argument("--name", default=None, help="count only: model name for outputs and readouts")
    ap.add_argument("--readouts", type=Path, default=None,
                    help="count only: save every final seed's head and the PCA under this directory")
    ap.add_argument("--features-device", default="cpu",
                    help="count only: hold the cell's features here, e.g. cuda")
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STAGE:LAYER",
        help="run only the given STAGE:LAYER cells",
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        choices=("tower", "merged", "projected"),
        default=None,
        help="run only these Stages",
    )
    ap.add_argument(
        "--deepest-only",
        action="store_true",
        help="run only cells at the final Tower layer",
    )
    args = ap.parse_args()
    if args.learning_rates is None:
        args.learning_rates = list(COUNT_LEARNING_RATES if args.task == "count" else LEARNING_RATES)
    if args.task == "count":
        count_main(args)
        return
    args.max_depth = depth_range(args.targets, args.max_depth)
    scenes = manifest(args.targets).get("scenes", {})
    print(f"depth bins span 0 to {args.max_depth:.1f} m")

    try:
        chosen = selected_slices(args.run, args.only, args.stages, args.deepest_only)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    cells = []
    order, targets, valid, coverage = None, None, None, None
    for stage, layer in chosen:
        started = time.perf_counter()
        batch = cache.load_batch(args.run, stage, layer)
        # Reuse the 6.3 GB target array while image order stays unchanged.
        if batch.image_ids != order:
            order = batch.image_ids
            targets, valid = load_targets(args.targets, order, args.task)
            coverage = coverage_of(targets, valid)
        split = split_indices(len(order))

        if args.match_capacity:
            batch = match_capacity(batch, split, args.width)
        # The cache holds bfloat16; the heads are float32.
        features = geometry.dense_map(batch).float()

        parameters = geometry.parameter_count(build_head(features.shape[1], args.task, args))
        result = run_cell(
            features, targets, valid, split, args.task, coverage, order, scenes, args
        )
        result |= {
            "max_depth": args.max_depth,
            "model_id": batch.model_id,
            "stage": stage,
            "layer_index": layer,
            "relative_depth": batch.relative_depth,
            "token_width": features.shape[1],
            "grid": features.shape[-1],
            "trainable_parameters": parameters,
            "seconds": round(time.perf_counter() - started, 1),
        }
        cells.append(result)
        headline = SELECTION[args.task][0]
        print(
            f"{stage:>9} L{layer:02d}  depth {result['relative_depth']:.3f}  "
            f"{headline} {result[headline]:.4f} +- {result[headline + '_std']:.4f}  "
            f"lr {result['learning_rate']:g}  params {parameters:,}  "
            f"{result['seconds']:.0f}s",
            flush=True,
        )

    tag = "matched" if args.match_capacity else "raw"
    out = args.out or args.run / f"geometry_{args.task}_{args.head}_{tag}.json"

    out.write_text(
        json.dumps(
            {"task": args.task, "images": len(order), "train_images": len(split.train),
             "capacity_matched": args.match_capacity, "head": args.head,
             "epochs": args.epochs, "seeds": args.seeds,
             "learning_rates": args.learning_rates, "cells": cells},
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
