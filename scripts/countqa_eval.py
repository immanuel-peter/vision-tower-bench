#!/usr/bin/env python3
"""Frozen-head transfer evaluation of the FSC-147 count readouts on curated CountQA.

This measures exemplar-based frozen-feature counting on a curated CountQA subset. It is not
MLLM question answering; never put these numbers beside MLLM CountQA accuracies.

Nothing is fit on CountQA. Each saved readout (head, plus PCA for the matched arm) is loaded
through ``vtb.geometry_run.load_readout`` and applied to the CountQA feature cache as is:

    tokens (cache, keyed by image_id) -> Readout.density(tokens, boxes, valid_rects)
    -> (N, 1, 64, 64) density -> counting.masked_count(density, counting.coverage(frame.valid, 64))

where ``frame`` is the letterbox frame from data/countqa/manifest.json and
``valid_rects = [frame.valid]``, which the readout maps onto its patch grid. Padding
never counts.

Primary result: the exemplar-conditioned arm (one approved human box per image), Tower Stage.
Secondary: the exemplar-free arm at every Stage. The conditioned arm refuses to run unless
results/counting/countqa/exemplars.json has status "approved".

``--annotation-source ai`` is a separate, experimental protocol (countqa-ai-exemplars-v1). It reads
the AI-authored boxes in results/counting/countqa/exemplars_ai.json (scripts/countqa_ai_exemplars.py),
keeps their provenance (annotation_source=ai, human_reviewed=false, approved_by=null), excludes the
no_clean_instance and unresolved needs_review questions, and scores both arms on exactly the
AI-boxed image IDs so conditioned and exemplar-free cells pair image for image. Outputs go to
results/counting/countqa/ai_exemplars/. The exemplar-free score on all kept questions comes only from
the default mode and stays separate.

Every CountQA result here is experimental and supplementary: an FSC-147-trained counting transfer on
a CountQA-derived subset, not official CountQA benchmark scores.

Metrics per Stage cell: MAE, RMSE (sqrt of the mean squared error over all seed x image rows),
exact match after floor(pred + 0.5), and per-bin values with sample counts (counting.count_bin
of the true answer). Stage differences use the paired image bootstrap of
vtb.geometry_bootstrap: seeds are paired by seed index, each image's metric is averaged over
seeds, images are resampled with replacement (10,000 resamples, bootstrap seed 0) and the
interval is the percentile 95% interval.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
from collections.abc import Callable
from math import isqrt
from pathlib import Path

import numpy as np
import torch

from scripts import countqa_ai_exemplars as ai_exemplars
from vtb import cache, counting
from vtb.geometry_bootstrap import paired_metric_bootstrap

LABEL = ("experimental, supplementary: exemplar-based frozen-feature counting transfer on a curated "
         "CountQA-derived subset; not official CountQA benchmark scores")
AI_LABEL = ai_exemplars.EVALUATION_LABEL
NOT_MLLM = ("Frozen Tower/Stage features read by FSC-147-trained count heads; not MLLM question "
            "answering and not comparable to MLLM CountQA scores.")
STAGE_ORDER = ("tower", "merged", "projected")
ARMS = ("exemplar", "free")
VARIANTS = ("raw", "matched")
TSV_COLUMNS = ("image_id", "split", "stage", "layer", "arm", "seed", "true_count", "predicted_count",
               "signed_error", "abs_error", "sq_error", "bin")
RESULTS = Path("results/counting/countqa")
AI_RESULTS = RESULTS / "ai_exemplars"
CELL_DIR = re.compile(r"^(?P<stage>tower|merged|projected)_L(?P<layer>\d+)$")


class Refused(RuntimeError):
    """The requested evaluation would break the CountQA protocol."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- inputs
def load_frames(manifest: dict) -> dict[str, counting.SquareFrame]:
    frames = {}
    for image_id, entry in manifest["images"].items():
        frame = counting.letterbox_frame(entry["width"], entry["height"])
        if frame.as_dict() != entry["frame"]:
            raise ValueError(f"{image_id}: manifest frame differs from counting.letterbox_frame")
        frames[image_id] = frame
    return frames


def load_exemplars(path: Path, manifest: dict) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Approved letterbox unit boxes per image ID and the excluded IDs with reasons.

    Raises Refused unless the file is approved, matches the frozen eligibility, every question is
    resolved, and every box is a human box consistent with its pixel coordinates.
    """
    if path is None or not Path(path).exists():
        raise Refused("conditioned evaluation needs results/counting/countqa/exemplars.json")
    data = json.loads(Path(path).read_text())
    if data.get("status") != "approved":
        raise Refused(f"exemplar annotations are {data.get('status')!r}, not 'approved'; "
                      "conditioned CountQA evaluation refuses unapproved boxes")
    if data.get("eligibility_sha256") != manifest["eligibility_sha256"]:
        raise Refused("exemplars were drawn for a different eligibility file")
    if not data.get("approved_by"):
        raise Refused("approved exemplars must name the human approver")
    boxes, excluded = {}, {}
    wanted = {entry["question_id"]: image_id for image_id, entry in manifest["images"].items()}
    if set(data["items"]) != set(wanted):
        raise Refused("exemplar questions differ from the prepared CountQA images")
    for question_id, item in data["items"].items():
        image_id = wanted[question_id]
        if item["status"] == "no_clean_instance":
            excluded[image_id] = "no_clean_instance"
            continue
        if item["status"] != "approved" or item.get("box_source") != "human" or item.get("box_px") is None:
            raise Refused(f"{question_id}: status {item['status']!r}, source {item.get('box_source')!r}")
        entry = manifest["images"][image_id]
        frame = counting.letterbox_frame(entry["width"], entry["height"])
        unit, kept = counting.clip_box(frame.boxes(np.asarray(item["box_px"], dtype=np.float64))[0])
        if kept < 1.0 - 1e-9 or not np.allclose(unit, item["box_letterbox_unit"], atol=1e-5):
            raise Refused(f"{question_id}: stored letterbox box does not match its pixel box")
        boxes[image_id] = unit[None, :].astype(np.float64)
    return boxes, excluded


def load_ai_exemplars(path: Path, manifest: dict) -> tuple[dict[str, np.ndarray], dict[str, str], dict]:
    """AI-authored letterbox unit boxes per image ID, the excluded IDs with reasons, and provenance.

    A separate protocol from load_exemplars: it accepts only the exemplars_ai.json format, keeps
    every record AI-sourced and unreviewed, and excludes no_clean_instance and needs_review rows.
    """
    if path is None or not Path(path).exists():
        raise Refused("AI-exemplar evaluation needs results/counting/countqa/exemplars_ai.json")
    data = json.loads(Path(path).read_text())
    if data.get("format") != ai_exemplars.AI_FORMAT or data.get("status") != ai_exemplars.AI_STATUS:
        raise Refused(f"{path} is not an {ai_exemplars.AI_FORMAT} file with status {ai_exemplars.AI_STATUS!r}")
    if data.get("eligibility_sha256") != manifest["eligibility_sha256"]:
        raise Refused("exemplars were drawn for a different eligibility file")
    wanted = {entry["question_id"]: image_id for image_id, entry in manifest["images"].items()}
    if set(data["items"]) != set(wanted):
        raise Refused("exemplar questions differ from the prepared CountQA images")
    records = [data, *data["items"].values()]
    if any(r.get("annotation_source") != "ai" or r.get("human_reviewed") is not False
           or r.get("approved_by") is not None for r in records):
        raise Refused("AI exemplars must keep annotation_source='ai', human_reviewed=false, approved_by=null")
    boxes, excluded = {}, {}
    for question_id, item in data["items"].items():
        image_id = wanted[question_id]
        if item["status"] in ai_exemplars.EXCLUSION:
            if item.get("box_px") is not None:
                raise Refused(f"{question_id}: {item['status']} row carries a box")
            excluded[image_id] = ai_exemplars.EXCLUSION[item["status"]]
            continue
        if item["status"] != "boxed" or item.get("box_source") != "ai" or item.get("box_px") is None:
            raise Refused(f"{question_id}: status {item['status']!r}, source {item.get('box_source')!r}")
        entry = manifest["images"][image_id]
        frame = counting.letterbox_frame(entry["width"], entry["height"])
        unit, kept = counting.clip_box(frame.boxes(np.asarray(item["box_px"], dtype=np.float64))[0])
        if kept < 1.0 - 1e-9 or not np.allclose(unit, item["box_letterbox_unit"], atol=1e-5):
            raise Refused(f"{question_id}: stored letterbox box does not match its pixel box")
        boxes[image_id] = unit[None, :].astype(np.float64)
    reasons = list(excluded.values())
    provenance = {
        "protocol": data["protocol"], "annotation_protocol": data["annotation_protocol"],
        "annotation_source": "ai", "annotation_model": data["annotation_model"], "human_reviewed": False,
        "approved_by": None, "annotation_revision": data.get("annotation_revision"),
        "exemplars_sha256": sha256_file(Path(path)), "source_export_sha256": data["source"]["export_sha256"],
        "review_coverage": {k: v for k, v in data["review_coverage"].items() if k != "statement"},
        "selection": {"questions": len(data["items"]), "included_boxed": len(boxes),
                      "excluded_no_clean_instance": reasons.count("no_clean_instance"),
                      "excluded_needs_review_unresolved": reasons.count("needs_review_unresolved")},
        "included_image_ids_sha256": hashlib.sha256("\n".join(sorted(boxes)).encode()).hexdigest(),
        "limitations": data["limitations"],
    }
    return boxes, excluded, provenance


def readout_cells(root: Path, arm: str, variant: str, stages: list[str] | None) -> list[tuple[str, int, Path]]:
    base = Path(root) / f"{arm}_{variant}"
    cells = []
    for path in sorted(base.iterdir()) if base.is_dir() else []:
        match = CELL_DIR.match(path.name)
        if match and (not stages or match["stage"] in stages):
            cells.append((match["stage"], int(match["layer"]), path))
    if not cells:
        raise FileNotFoundError(f"no <stage>_L<layer> readouts under {base}")
    return sorted(cells, key=lambda c: (STAGE_ORDER.index(c[0]), c[1]))


def check_cache(features: Path, manifest_path: Path, allow_incomplete: bool) -> dict:
    complete = Path(features) / "complete.json"
    if not complete.exists():
        if allow_incomplete:
            return {"complete_json": None, "allow_incomplete_cache": True}
        raise Refused(f"{complete} is missing; a cache without a valid complete.json is incomplete")
    info = json.loads(complete.read_text())
    expected = sha256_file(manifest_path)
    if info.get("manifest_sha256") != expected:
        raise Refused(f"{complete}: manifest_sha256 does not match {manifest_path}")
    return {"complete_json": str(complete), "complete_json_sha256": sha256_file(complete)}


def default_loader() -> Callable:
    from vtb import geometry_run

    loader = getattr(geometry_run, "load_readout", None)
    if loader is None:
        raise RuntimeError("vtb.geometry_run.load_readout is not implemented yet (lead)")
    return loader


# ---------------------------------------------------------------- frozen state
def frozen_state(readout) -> str:
    """sha256 over the head's state dict, the reducer tensors, the arm flag and train mode."""
    digest = hashlib.sha256()
    for name, tensor in sorted(readout.head.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                      if tensor.numel() else b"")
    if readout.reducer is not None:
        for name in ("mean", "basis"):
            tensor = getattr(readout.reducer, name)
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    digest.update(f"conditioned={readout.conditioned};training={readout.head.training}".encode())
    return digest.hexdigest()


# ---------------------------------------------------------------- inference
@torch.inference_mode()
def predict_counts(readout, tokens: torch.Tensor, image_ids: list[str], frames: dict,
                   boxes: dict[str, np.ndarray] | None, batch_size: int, device: str) -> np.ndarray:
    """Masked counts for ``image_ids`` (rows of ``tokens`` in the same order)."""
    if readout.head.training:
        raise Refused("readout head is in training mode")
    side = isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise ValueError(f"{tokens.shape[1]} tokens do not form a square grid")
    if readout.conditioned and boxes is None:
        raise Refused("conditioned readout without approved exemplar boxes")
    counts = []
    for start in range(0, len(image_ids), batch_size):
        chunk = image_ids[start:start + batch_size]
        valid_target = torch.from_numpy(np.stack([counting.coverage(frames[i].valid, counting.TARGET_GRID)
                                                  for i in chunk]))
        chunk_boxes = [boxes[i] for i in chunk] if readout.conditioned else None
        # The readout maps each valid rectangle onto its own patch grid, which may span past the canvas.
        density = readout.density(tokens[start:start + len(chunk)].to(device), boxes=chunk_boxes,
                                  valid_rects=[frames[i].valid for i in chunk])
        if density.shape != (len(chunk), 1, counting.TARGET_GRID, counting.TARGET_GRID):
            raise ValueError(f"density shape {tuple(density.shape)}")
        counts.append(counting.masked_count(density.float().cpu(), valid_target))
    return torch.cat(counts).numpy().astype(np.float64)


# ---------------------------------------------------------------- metrics
def exact(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Round half up, floor(pred + 0.5), compared with the integer answer."""
    return (np.floor(predicted + 0.5) == true).astype(np.float64)


def summarise(predicted: np.ndarray, true: np.ndarray) -> dict:
    """predicted: (seeds, images); true: (images,)."""
    error = predicted - true[None, :]
    correct = exact(predicted, true[None, :])
    bins = np.array([counting.count_bin(value) for value in true])
    out = {
        "images": int(len(true)), "seeds": int(predicted.shape[0]),
        # Same convention as the FSC-147 cells: each seed's RMSE, then the mean over seeds;
        # rmse_pooled is sqrt of the mean squared error over all seed x image rows.
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean(axis=1)).mean()),
        "rmse_pooled": float(np.sqrt((error ** 2).mean())),
        "exact_match": float(correct.mean()), "mean_signed_error": float(error.mean()),
        "mae_by_seed": np.abs(error).mean(axis=1).tolist(),
        "rmse_by_seed": np.sqrt((error ** 2).mean(axis=1)).tolist(),
        "exact_match_by_seed": correct.mean(axis=1).tolist(),
        "by_bin": {},
    }
    for name in counting.BINS:
        mask = bins == name
        out["by_bin"][name] = {"images": int(mask.sum())} | ({
            "mae": float(np.abs(error[:, mask]).mean()),
            "rmse": float(np.sqrt((error[:, mask] ** 2).mean(axis=1)).mean()),
            "exact_match": float(correct[:, mask].mean()),
        } if mask.any() else {})
    return out


def stage_differences(cells: list[dict], true: np.ndarray, resamples: int, confidence: float,
                      seed: int) -> dict:
    """Paired image bootstrap between every two Stage cells on identical IDs and seeds."""
    out = {}
    for first, second in itertools.combinations(cells, 2):
        if first["image_ids"] != second["image_ids"] or first["seeds"] != second["seeds"]:
            raise ValueError("stage cells must share image IDs and seeds for pairing")
        key = f"{first['key']}_vs_{second['key']}"
        a, b = torch.from_numpy(first["predicted"]), torch.from_numpy(second["predicted"])
        truth = torch.from_numpy(true)[None, :]
        out[key] = {
            "first": first["key"], "second": second["key"], "images": len(first["image_ids"]),
            "seeds_paired": first["seeds"],
            "abs_error": paired_metric_bootstrap((a - truth).abs(), (b - truth).abs(), higher_is_better=False,
                                                 resamples=resamples, confidence=confidence, seed=seed),
            "exact_match": paired_metric_bootstrap(
                torch.from_numpy(exact(first["predicted"], true[None, :])),
                torch.from_numpy(exact(second["predicted"], true[None, :])),
                higher_is_better=True, resamples=resamples, confidence=confidence, seed=seed),
        }
    return out


def prediction_rows(cell: dict, true: np.ndarray, arm: str) -> list[list]:
    rows = []
    for s, seed in enumerate(cell["seeds"]):
        for i, image_id in enumerate(cell["image_ids"]):
            pred, truth = float(cell["predicted"][s, i]), float(true[i])
            error = pred - truth
            rows.append([image_id, "countqa", cell["stage"], cell["layer"], arm, seed, int(truth),
                         f"{pred:.6f}", f"{error:.6f}", f"{abs(error):.6f}", f"{error * error:.6f}",
                         counting.count_bin(truth)])
    return rows


# ---------------------------------------------------------------- driver
def evaluate(args, loader: Callable | None = None) -> dict:
    """Run the evaluation and return the payload (written by main)."""
    manifest = json.loads(Path(args.manifest).read_text())
    frames = load_frames(manifest)
    ai = getattr(args, "annotation_source", "human") == "ai"
    exemplar_status = (json.loads(Path(args.exemplars).read_text()).get("status")
                       if args.exemplars and Path(args.exemplars).exists() else None)
    boxes, excluded, annotation = None, {}, None
    if ai:
        # Both arms run on the AI-boxed IDs only; refuses before any readout loads.
        boxes, excluded, annotation = load_ai_exemplars(args.exemplars, manifest)
    elif args.arm == "exemplar":
        boxes, excluded = load_exemplars(args.exemplars, manifest)  # refuses before any readout loads
    elif exemplar_status == "approved":
        boxes, excluded = load_exemplars(args.exemplars, manifest)  # common-subset report only
    cache_info = check_cache(args.features, Path(args.manifest), args.allow_incomplete_cache)
    loader = loader or default_loader()

    image_ids = sorted(i for i in manifest["images"] if not ((ai or args.arm == "exemplar") and i in excluded))
    true = np.array([manifest["images"][i]["answer"] for i in image_ids], dtype=np.float64)
    cells = []
    for stage, layer, directory in readout_cells(args.readouts, args.arm, args.variant, args.stages):
        tokens, cache_ids, meta = cache.load(args.features, stage, layer)
        if len(set(cache_ids)) != len(cache_ids):
            raise ValueError(f"duplicate image IDs in the {stage} L{layer} cache")
        if set(cache_ids) != set(manifest["images"]):
            raise ValueError(f"{stage} L{layer} cache IDs differ from the CountQA manifest")
        position = {image_id: k for k, image_id in enumerate(cache_ids)}
        ordered = tokens[torch.tensor([position[i] for i in image_ids])]
        predicted, seeds_meta = [], []
        for seed in args.seeds:
            readout = loader(directory, seed, args.device)
            if readout.conditioned != (args.arm == "exemplar"):
                raise Refused(f"{directory} seed {seed}: conditioned={readout.conditioned} but arm {args.arm}")
            if readout.conditioned and boxes is None:
                raise Refused("conditioned readout without approved exemplar boxes")
            if (readout.reducer is not None) != (args.variant == "matched"):
                raise Refused(f"{directory}: reducer presence does not match the {args.variant} arm")
            protocol = readout.meta.get("protocol")
            if protocol not in (None, counting.PROTOCOL):
                raise Refused(f"{directory}: protocol {protocol!r} is not {counting.PROTOCOL}")
            expected = {"model_id": meta.get("model_id"), "stage": stage, "layer": layer,
                        "resolution": int(meta["resolution"]), "tokens": int(tokens.shape[1])}
            found = {key: readout.meta.get(key) for key in expected}
            if found != expected:
                raise Refused(f"{directory} seed {seed}: readout {found} does not match cache {expected}")
            before = frozen_state(readout)
            predicted.append(predict_counts(readout, ordered, image_ids, frames, boxes,
                                            args.batch_size, args.device))
            after = frozen_state(readout)
            if after != before:
                raise AssertionError(f"{directory} seed {seed}: readout state changed during evaluation")
            seeds_meta.append({"seed": seed, "frozen_state_sha256": before,
                               "meta": {k: readout.meta[k] for k in sorted(readout.meta)
                                        if isinstance(readout.meta[k], (str, int, float, bool))}})
        predicted = np.stack(predicted)
        cell = {"key": directory.name,"stage": stage, "layer": layer, "readout_dir": str(directory),
                "model_id": meta.get("model_id"), "relative_depth": meta.get("relative_depth"),
                "tokens_per_image": int(tokens.shape[1]), "seeds": list(args.seeds),
                "image_ids": image_ids, "predicted": predicted}
        cell["metrics"] = summarise(predicted, true)
        if args.arm == "free" and boxes is not None and not ai:
            common = [k for k, i in enumerate(image_ids) if i not in excluded]
            cell["metrics_common_exemplar_subset"] = summarise(predicted[:, common], true[common])
        cell["seed_readouts"] = seeds_meta
        cells.append(cell)
        del tokens, ordered

    differences = stage_differences(cells, true, args.resamples, args.confidence, args.bootstrap_seed)
    primary = next((c["key"] for c in cells if c["stage"] == "tower"), None) if args.arm == "exemplar" else None
    if ai:
        role = ("experimental, supplementary: exemplar-conditioned, one AI-authored box (not human-reviewed)"
                if args.arm == "exemplar" else
                "experimental, supplementary: exemplar-free on the AI-boxed IDs, paired with the conditioned arm")
    else:
        role = ("primary (exemplar-conditioned, one approved human box)" if args.arm == "exemplar"
                else "secondary (exemplar-free)")
    payload = {
        "label": AI_LABEL if ai else LABEL,
        "experimental": True, "supplementary": True, "official_countqa_benchmark": False,
        "not_mllm_qa": NOT_MLLM,
        "role": role,
        "annotation": annotation,
        "primary_cell": primary,
        "model": Path(args.readouts).name,
        "arm": args.arm,
        "variant": args.variant,
        "dataset": {
            "name": "CountQA (curated subset)", "repo": manifest["source"]["repo"],
            "revision": manifest["source"]["revision"], "license": manifest["source"]["license"],
            "citation": manifest["source"]["citation"],
            "eligibility_sha256": manifest["eligibility_sha256"],
            "manifest_sha256": sha256_file(Path(args.manifest)),
            "exemplars_status": exemplar_status,
            "exemplars_sha256": sha256_file(Path(args.exemplars)) if exemplar_status else None,
            "prepared_images": len(manifest["images"]), "evaluated_images": len(image_ids),
            "excluded": {i: r for i, r in sorted(excluded.items())} if (ai or args.arm == "exemplar") else {},
        },
        "features": {"dir": str(args.features), **cache_info},
        "protocol": {
            "training": "none on CountQA: saved FSC-147 heads and PCA are applied frozen",
            "frame": "letterbox (counting.letterbox_frame), whole image, fill counting.LETTERBOX_FILL",
            "transfer_preprocessing": manifest["transfer_preprocessing"],
            "count": "counting.masked_count(density, counting.coverage(frame.valid, 64))",
            "prototype_valid": "counting.token_valid(frame.valid, side, grid_extent) at the token grid",
            "exemplars": ("one AI-authored box per image (countqa-ai-exemplars-v1, not human-reviewed), "
                          "frame.boxes of the source-pixel box" if ai else
                          "one approved human box per image, frame.boxes of the source-pixel box")
            if args.arm == "exemplar" else
            ("none; scored on the AI-boxed image IDs only" if ai else "none"),
            "exact_match": "floor(pred + 0.5) == integer answer",
            "bins": "counting.count_bin of the true answer",
            "rmse": "mean over seeds of each seed's sqrt(mean sq_error); rmse_pooled pools seed x image rows",
            "numerics": "readouts trained with cuDNN TF32 convolutions on GPU; this run's device may differ, "
                        "so transfer parity with training is approximate at float32 rounding",
            "seeds": list(args.seeds),
            "bootstrap": {"function": "vtb.geometry_bootstrap.paired_metric_bootstrap",
                          "pairing": "identical image IDs; seed k with seed k; per-image seed-mean metric",
                          "resamples": args.resamples, "confidence": args.confidence,
                          "interval": "percentile", "bootstrap_seed": args.bootstrap_seed,
                          "abs_error_difference": "second_minus_first (positive: first has lower error)",
                          "exact_match_difference": "first_minus_second"},
        },
        "cells": [{k: v for k, v in cell.items() if k not in ("predicted", "image_ids")} for cell in cells],
        "stage_differences": differences,
        "_cells": cells,
        "_true": true,
    }
    return payload


def write(payload: dict, out: Path) -> None:
    cells, true = payload.pop("_cells"), payload.pop("_true")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    tsv = out.with_suffix(".predictions.tsv")
    with open(tsv, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for cell in cells:
            writer.writerows(prediction_rows(cell, true, payload["arm"]))
    print(f"wrote {out} and {tsv}")


def parser() -> argparse.ArgumentParser:
    work = Path(os.environ.get("VTB_WORK", Path.home() / "vtb-work" / "counting-v3"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True,
                    help="vtb.extract run dir for the CountQA letterbox images, e.g. "
                    "$VTB_WORK/features/countqa/<model>_<res>_full")
    ap.add_argument("--readouts", type=Path, required=True,
                    help="$VTB_WORK/readouts/count-v4/<model> (holds <arm>_<variant>/<stage>_L<layer>/)")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="matched")
    ap.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--manifest", type=Path, default=work / "data" / "countqa" / "manifest.json")
    ap.add_argument("--annotation-source", choices=("human", "ai"), default="human",
                    help="ai: the experimental AI-exemplar protocol (exemplars_ai.json, both arms on the "
                         "AI-boxed IDs, outputs under ai_exemplars/)")
    ap.add_argument("--exemplars", type=Path, default=None,
                    help="default results/counting/countqa/exemplars.json, or exemplars_ai.json with "
                         "--annotation-source ai")
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/counting/countqa/[ai_exemplars/]<model>_count_<arm>_<variant>.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--allow-incomplete-cache", action="store_true",
                    help="development only: run without complete.json (recorded in the output)")
    return ap


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    ai = args.annotation_source == "ai"
    if args.exemplars is None:
        args.exemplars = RESULTS / ("exemplars_ai.json" if ai else "exemplars.json")
    out = args.out or (AI_RESULTS if ai else RESULTS) / f"{args.readouts.name}_count_{args.arm}_{args.variant}.json"
    try:
        payload = evaluate(args)
    except Refused as error:
        raise SystemExit(f"refused: {error}") from None
    write(payload, out)


if __name__ == "__main__":
    main()
