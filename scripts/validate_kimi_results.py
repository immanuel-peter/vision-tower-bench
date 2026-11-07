"""Validate correspondence payload contents and source-backed SPair exclusions."""

import argparse
import hashlib
import json
import math
from pathlib import Path

STAGES = {"tower:27", "merged:27", "projected:27"}
PRIMARY = {"navi": "recall_2cm", "scannet": "recall_10px", "spair": "pck_0.1"}
DEPTH_SLICES = {("tower", layer) for layer in (3, 7, 10, 14, 17, 20, 24, 27)} | {("merged", 27), ("projected", 27)}
SEMANTIC_RATES = {1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1, .3, 1.0}
GEOMETRY_RATES = {.0001, .0003, .001, .003, .01, .03}


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def validate_perturbation(payload, model_id, selected_ids, conditions, expected_splits,
                          release_revision, transform_sha256, device="cpu", threads=4):
    if len(selected_ids) != 2000 or len(set(selected_ids)) != 2000 or payload["images"] != 2000:
        raise ValueError("perturbation requires all 2000 selected image IDs")
    if len(conditions) != 13 or len(set(conditions)) != 13 or "identity" not in conditions:
        raise ValueError("perturbation requires exactly thirteen source conditions")
    if not finite_numbers(payload) or payload["transform_manifest_sha256"] != transform_sha256:
        raise ValueError("nonfinite perturbation result or changed transform manifest")
    protocol = payload["protocol"]
    if (protocol["readout"] != "attention" or protocol["classes"] != 100 or protocol["capacity_matched_width"] != 512
            or protocol["epochs"] != 20 or protocol["seeds"] != 3
            or set(protocol["learning_rates"]) != SEMANTIC_RATES
            or protocol["learning_rate_search_seed"] != 0 or protocol["retraining_seeds"] != [0, 1, 2]
            or protocol["device"] != device or protocol["torch_threads"] != threads
            or protocol["batch_size"] != 256 or protocol["standardize_channels"]):
        raise ValueError("perturbation readout/search/seed/device protocol differs")
    if set(payload["feature_provenance"]) != set(conditions):
        raise ValueError("missing perturbation feature provenance")
    for condition in conditions:
        feature = payload["feature_provenance"][condition]["details"]
        if (feature["model_id"] != model_id or feature["release_revision"] != release_revision
                or feature["resolution"] != 448 or feature["pooled_to"] != 4
                or feature["batch_size"] != 1 or feature["dtype"] != "bfloat16"
                or feature["device"] != "cuda:0" or feature["physical_gpu"] != 1
                or feature["threads"] != 8 or feature["workers"] != 2
                or feature["condition"] != condition
                or feature["ordered_ids_sha256"] != ids_digest(selected_ids)):
            raise ValueError("perturbation feature checkpoint/producer/IDs differ")
    cells = payload["cells"]
    if len(cells) != 3 or {(cell["stage"], cell["layer_index"]) for cell in cells} != {(stage, 27) for stage in ("tower", "merged", "projected")}:
        raise ValueError("perturbation requires all three final Stages")
    expected_hashes = {name: ids_digest(values) for name, values in expected_splits.items()}
    for cell in cells:
        if (cell["model_id"] != model_id or cell["seed_ids"] != [0, 1, 2]
                or cell["relative_depth"] != 1.0 or not cell["capacity_matched"] or cell["readout"] != "attention"
                or cell["learning_rate_search_seed"] != 0
                or cell["sample_ids_sha256"] != ids_digest(selected_ids)
                or cell["split_ids_sha256"] != expected_hashes
                or any(cell[f"{name}_images"] != len(expected_splits[name]) for name in ("train", "test"))
                or cell["validation_images"] != len(expected_splits["val"])):
            raise ValueError("perturbation actual sample/split/seed evidence differs")
        search = cell["learning_rate_search"]
        if len(search) != 11 or {float(rate) for rate in search} != SEMANTIC_RATES:
            raise ValueError("perturbation learning-rate grid differs")
        chosen = [value for rate, value in search.items() if float(rate) == cell["learning_rate"]]
        if len(chosen) != 1 or chosen[0] != max(search.values()) or round(cell["validation_accuracy"], 4) != chosen[0]:
            raise ValueError("perturbation selected learning rate differs from validation scores")
        values = cell["conditions"]
        if len(values) != 13 or {row["condition"] for row in values} != set(conditions):
            raise ValueError("perturbation condition cells differ from source manifest")
        if set(cell["measurements"]) != set(conditions) - {"identity"}:
            raise ValueError("perturbation paired measurement conditions differ")
        clean = None
        for condition, rows in cell["measurements"].items():
            if [row["image_id"] for row in rows] != expected_splits["test"]:
                raise ValueError("perturbation actual paired test IDs/order differ")
            for row in rows:
                if any(len(row[key]) != 3 or any(value not in (0., 1.) for value in row[key]) for key in ("clean", "perturbed")):
                    raise ValueError("perturbation correctness must contain three binary seed values")
            current_clean = [row["clean"] for row in rows]
            if clean is not None and current_clean != clean:
                raise ValueError("clean correctness changed between perturbation conditions")
            clean = current_clean
            result = next(row for row in values if row["condition"] == condition)
            seed_scores = [sum(row["perturbed"][seed] for row in rows) / len(rows) for seed in range(3)]
            if len(result["accuracy_by_seed"]) != 3 or any(abs(x - y) > 1e-6 for x, y in zip(seed_scores, result["accuracy_by_seed"], strict=True)):
                raise ValueError("perturbation seed accuracy differs from actual paired correctness")
            if abs(result["accuracy"] - sum(seed_scores) / 3) > 1e-6:
                raise ValueError("perturbation accuracy differs from actual paired correctness")
            interval = result["clean_minus_condition"]
            point = sum(sum(row["clean"]) - sum(row["perturbed"]) for row in rows) / (3 * len(rows))
            if (abs(interval["point_estimate"] - point) > 1e-6 or interval["lower"] > interval["upper"]
                    or interval["resamples"] != 10000 or interval["bootstrap_seed"] != 0 or interval["confidence"] != .95):
                raise ValueError("perturbation interval differs from same-run paired measurements")
        identity = next(row for row in values if row["condition"] == "identity")
        clean_scores = [sum(row[seed] for row in clean) / len(clean) for seed in range(3)]
        if len(identity["accuracy_by_seed"]) != 3 or any(abs(x - y) > 1e-6 for x, y in zip(clean_scores, identity["accuracy_by_seed"], strict=True)) or abs(identity["accuracy"] - sum(clean_scores) / 3) > 1e-6:
            raise ValueError("identity accuracy differs from actual clean paired correctness")
    return dict(passed=True, images=2000, stages=3, conditions=13, cells=39,
                test_images=len(expected_splits["test"]), learning_rates=11, search_seed=0,
                retraining_seeds=[0, 1, 2], fit_device=device, fit_threads=threads,
                feature_device="cuda:0", feature_physical_gpu=1)


def validate_probe(payload, model_id, selected_ids, kind, arm, readout="attention", label_fraction=1.0, expected_splits=None, expected_slices=None):
    semantic = kind in {"recognition", "label-budget", "standardized"}
    expected_count = 13000 if semantic else (1000 if kind == "kitti" else 771)
    if len(selected_ids) != expected_count or len(set(selected_ids)) != expected_count or payload["images"] != expected_count:
        raise ValueError("primary sample IDs/count do not cover the full prescribed dataset")
    expected = DEPTH_SLICES
    if kind in {"label-budget", "standardized"}:
        expected = {("tower", 27)}
    elif kind == "kitti":
        expected = {("tower", 27), ("projected", 27)}
    if expected_slices is not None:
        if kind != "recognition" or not expected_slices or not set(expected_slices) <= DEPTH_SLICES:
            raise ValueError("invalid explicitly scoped recognition slices")
        expected = set(expected_slices)
    cells = payload["cells"]
    actual = [(cell["stage"], cell["layer_index"]) for cell in cells]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("actual primary Stage/depth cells do not equal inventory")
    if not finite_numbers(payload):
        raise ValueError("nonfinite primary metric or learning-rate score")
    rates = SEMANTIC_RATES if semantic else GEOMETRY_RATES
    feature = payload["feature_provenance"]["details"]
    if feature["model_id"] != model_id or feature["resolution"] != 448 or feature["batch_size"] != 1:
        raise ValueError("feature extraction checkpoint/resolution/batch provenance differs")
    if feature["dtype"].removeprefix("torch.") != "bfloat16" or feature["pooled_to"] != (4 if semantic else None):
        raise ValueError("feature dtype/pooling provenance differs")
    if semantic:
        protocol = payload["protocol"]
        if (protocol["epochs"] != 20 or protocol["readout"] != readout or protocol["classes"] != 100
                or protocol["capacity_matched"] != (arm == "matched") or protocol["width"] != 512
                or set(protocol["learning_rates"]) != rates or protocol["retraining_seeds"] != [0, 1, 2]):
            raise ValueError("semantic training protocol differs from requested cell")
    if expected_splits is None or payload["train_images"] != len(expected_splits["train"]):
        raise ValueError("actual training-image count differs from audited split")
    if semantic and (payload["standardize_channels"] != (kind == "standardized") or
                     payload["full_train_images"] != expected_count - 2 * int(expected_count * .15)):
        raise ValueError("primary/diagnostic mode or full training count differs")
    for cell in cells:
        if cell["model_id"] != model_id or cell["sample_ids_sha256"] != ids_digest(selected_ids):
            raise ValueError("primary checkpoint or actual ordered sample IDs differ")
        if cell["seeds"] != 3 or cell["seed_ids"] != [0, 1, 2] or cell["learning_rate_search_seed"] != 0:
            raise ValueError("learning-rate search/retraining seed protocol differs")
        searched = cell["learning_rate_search"]
        if {float(rate) for rate in searched} != rates or len(searched) != len(rates):
            raise ValueError("learning-rate grid differs from corrected protocol")
        selected_rate = cell["learning_rate"]
        if selected_rate not in rates:
            raise ValueError("selected learning rate was not searched")
        selected_score = next(score for rate, score in searched.items() if float(rate) == selected_rate)
        higher = semantic or payload["task"] == "depth"
        best_score = max(searched.values()) if higher else min(searched.values())
        if selected_score != best_score:
            raise ValueError("selected learning rate is inconsistent with validation scores")
        if abs(cell["relative_depth"] - cell["layer_index"] / 27) > 1e-4:
            raise ValueError("relative depth does not match checkpoint depth")
        if set(cell["split_ids_sha256"]) != {"train", "val", "test"}:
            raise ValueError("missing actual split identity evidence")
        if expected_splits is None or cell["split_ids_sha256"] != {name: ids_digest(expected_splits[name]) for name in ("train", "val", "test")}:
            raise ValueError("actual split identities differ from audited seed-0 split")
        if semantic:
            scores = cell["test_accuracy_by_seed"]
            if len(scores) != 3 or any(not 0 <= value <= 1 for value in scores):
                raise ValueError("missing or invalid actual seed scores")
            if abs(cell["test_accuracy"] - sum(scores) / 3) > 1e-4:
                raise ValueError("reported accuracy differs from its actual seed scores")
            if cell["label_fraction"] != label_fraction or payload["label_fraction"] != label_fraction:
                raise ValueError("label budget differs from expected cell")
            if cell["train_images"] != len(expected_splits["train"]):
                raise ValueError("cell training count differs from its actual split IDs")
            if cell["standardize_channels"] != (kind == "standardized"):
                raise ValueError("diagnostic standardization leaked into primary behavior")
            native_width = {"tower": 1024, "merged": 4096, "projected": 7168}[cell["stage"]]
            if cell["token_width"] != native_width:
                raise ValueError("native token width differs from Stage definition")
            fitted_width = 512 if arm == "matched" else native_width
            parameters = fitted_width * 100 + 100 if readout == "mean" else fitted_width * 1024 + 1103460
            if cell["trainable_parameters"] != parameters:
                raise ValueError("readout capacity differs from requested arm")
            if "test_predictions_by_seed" in cell:
                fit_device = protocol["device"]
                if fit_device == "per-cell":
                    source = cell.get("source_fit", {})
                    original = source.get("protocol", {})
                    if (kind != "recognition" or expected_slices is not None
                            or source.get("device") not in {"cpu", "cuda:0"}
                            or original.get("device") != source.get("device")
                            or {key: value for key, value in original.items() if key != "device"}
                            != {key: value for key, value in protocol.items() if key != "device"}):
                        raise ValueError("per-cell fit device requires an unchanged original recognition protocol")
                    fit_device = source["device"]
                predictions, labels = cell["test_predictions_by_seed"], cell["test_labels"]
                if (cell["test_image_ids"] != expected_splits["test"] or len(labels) != len(expected_splits["test"])
                        or len(predictions) != 3 or any(len(row) != len(labels) for row in predictions)
                        or any(type(value) is not int or not 0 <= value < 100 for row in [labels, *predictions] for value in row)):
                    raise ValueError("saved prediction IDs/labels/seeds differ from actual test split")
                actual = [sum(x == y for x, y in zip(row, labels, strict=True)) / len(labels) for row in predictions]
                if any(abs(x - y) > 1e-7 for x, y in zip(actual, scores, strict=True)):
                    raise ValueError("saved predictions disagree with actual seed accuracy")
                unrounded = cell["learning_rate_search_unrounded"]
                if (len(unrounded) != 11 or {float(rate) for rate in unrounded} != rates
                        or any(round(value, 4) != cell["learning_rate_search"][rate] for rate, value in unrounded.items())
                        or cell["validation_accuracy_unrounded"] != max(unrounded.values())
                        or next(value for rate, value in unrounded.items() if float(rate) == cell["learning_rate"]) != max(unrounded.values())
                        or cell["fit_device"] != fit_device):
                    raise ValueError("saved prediction fit/search provenance differs")
        else:
            if payload["epochs"] != 10 or payload["capacity_matched"] != (arm == "matched"):
                raise ValueError("geometry training protocol differs")
            if len(cell["per_seed"]) != 3 or any(seed["images"] != int(expected_count * .15) for seed in cell["per_seed"]):
                raise ValueError("missing geometry seed metrics or wrong test-image count")
            width = 512 if arm == "matched" else {"tower": 1024, "merged": 4096, "projected": 7168}[cell["stage"]]
            if cell["token_width"] != width or cell["grid"] != (32 if cell["stage"] == "tower" else 16):
                raise ValueError("geometry Stage width/grid differs from full-grid protocol")
    return dict(passed=True, images=expected_count, cells=len(cells), learning_rates=len(rates),
                feature_device=feature["device"],
                search_seed=0, retraining_seeds=[0, 1, 2], ordered_ids_sha256=ids_digest(selected_ids))


def validate_cache(run, model_id, selected_ids, kind):
    import torch
    from safetensors import safe_open
    from vtb import cache
    torch.set_num_threads(4)
    expected = {("tower", layer) for layer in (3, 7, 10, 14, 17, 20, 24, 27)} | {("merged", 27), ("projected", 27)}
    if kind == "perturbation":
        expected = {(stage, 27) for stage in ("tower", "merged", "projected")}
    elif kind == "kitti":
        expected = {("tower", 27), ("projected", 27)}
    if set(cache.slices(run)) != expected:
        raise ValueError("actual cache Stage/depth cells differ from inventory")
    widths = {"tower": 1024, "merged": 4096, "projected": 7168}
    pooled = kind in {"semantic", "perturbation"}
    records = []
    for stage, layer in sorted(expected):
        ids = []
        for path in cache.shards(run, stage, layer):
            with safe_open(path, framework="pt") as handle:
                meta = handle.metadata()
                chunk_ids = meta["image_ids"].split("\n")
                if meta["model_id"] != model_id or int(meta["resolution"]) != 448 or int(meta["num_layers"]) != 27:
                    raise ValueError(f"checkpoint/resolution metadata mismatch: {path}")
                if meta["stage"] != stage or int(meta["layer_index"]) != layer:
                    raise ValueError(f"Stage/depth metadata differs from shard filename: {path}")
                if meta["pooled_to"] != ("4" if pooled else "none"):
                    raise ValueError(f"pooling differs from task protocol: {path}")
                tensor = handle.get_tensor("tokens")
                count = 16 if pooled else (1024 if stage == "tower" else 256)
                if tuple(tensor.shape) != (len(chunk_ids), count, widths[stage]) or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"shape/dtype mismatch: {path}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"nonfinite cached tensor: {path}")
                ids.extend(chunk_ids)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append(dict(file=path.name, sha256=digest, stage=stage, layer=layer, images=len(chunk_ids)))
        if ids != selected_ids or len(ids) != len(set(ids)):
            raise ValueError(f"cache image IDs differ from selected manifest: {stage} {layer}")
    return dict(passed=True, slices=len(expected), images=len(selected_ids), shards=len(records), files=records)


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return not isinstance(value, (float, int)) or math.isfinite(value)


def validate_correspondence(payload, selected_ids, visibility=None):
    rows = payload["measurements"]
    if [row["pair_id"] for row in rows] != selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("actual selected pair IDs/order differ from the source manifest")
    if payload["pairs"] != len(selected_ids) or payload["resolution"] != 448:
        raise ValueError("pair count or resolution differs from protocol")
    if len(payload["cells"]) != 3 or {f"{c['stage']}:{c['layer_index']}" for c in payload["cells"]} != STAGES:
        raise ValueError("aggregate Stage cells differ from protocol")
    if not finite_numbers(payload["cells"]):
        raise ValueError("nonfinite aggregate metric")
    expected = {r["pair_id"]: r for r in visibility["measurements"]} if visibility else {}
    exclusions = []
    for row in rows:
        if set(row["cells"]) != STAGES:
            raise ValueError(f"missing or extra Stage: {row['pair_id']}")
        for stage, cell in row["cells"].items():
            if payload["dataset"] == "spair":
                source_count = expected[row["pair_id"]]["valid_keypoints"]
                if cell["keypoints"] != source_count:
                    raise ValueError(f"keypoint count differs from source: {row['pair_id']}")
                if source_count == 0:
                    if not math.isnan(cell["pck_0.1"]):
                        raise ValueError("empty pair must retain the protocol's NaN, not an invented score")
                    exclusions.append(dict(pair_id=row["pair_id"], stage=stage, reason="source-confirmed zero visible shared keypoints"))
                    continue
            if not finite_numbers(cell):
                raise ValueError(f"nonfinite metric on nonempty pair: {row['pair_id']} {stage}")
    return dict(selected_pairs=len(rows), exclusions=exclusions,
                nonempty_pairs=len(rows) - len({r["pair_id"] for r in exclusions}), passed=True)


def validate_pairing(first, second, stage, second_stage=None):
    if first["dataset"] != second["dataset"]:
        raise ValueError("paired datasets differ")
    left = {row["pair_id"]: row for row in first["measurements"]}
    right = {row["pair_id"]: row for row in second["measurements"]}
    if left.keys() != right.keys():
        raise ValueError("paired selection IDs differ")
    metric = PRIMARY[first["dataset"]]
    finite_ids, exclusions = [], []
    for pair_id in sorted(left):
        a, b = left[pair_id]["cells"][stage], right[pair_id]["cells"][second_stage or stage]
        a_finite, b_finite = math.isfinite(a[metric]), math.isfinite(b[metric])
        if a_finite and b_finite:
            finite_ids.append(pair_id)
        else:
            for cell in (a, b):
                if not math.isfinite(cell[metric]) and (first["dataset"] != "spair" or cell["keypoints"] != 0):
                    raise ValueError(f"unexplained nonfinite paired metric: {pair_id}")
            exclusions.append(dict(pair_id=pair_id, first_excluded=not a_finite, second_excluded=not b_finite,
                                   reason="source-confirmed zero keypoints, requiring individual payload audit"))
    return dict(stage=stage, second_stage=second_stage or stage, selected_pairs=len(left), finite_intersection=len(finite_ids),
                finite_ids=finite_ids, exclusion_union=exclusions,
                same_excluded_ids=all(r["first_excluded"] == r["second_excluded"] for r in exclusions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--visibility", type=Path)
    parser.add_argument("--cache-kind", choices=("semantic", "geometry", "kitti", "perturbation"))
    parser.add_argument("--model-id")
    parser.add_argument("--probe-kind", choices=("recognition", "geometry", "kitti", "label-budget", "standardized"))
    parser.add_argument("--arm", choices=("raw", "matched"), default="matched")
    parser.add_argument("--readout", choices=("attention", "mean"), default="attention")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--expected-splits", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--first-stage", choices=("tower", "merged", "projected"))
    parser.add_argument("--second-stage", choices=("tower", "merged", "projected"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ids = json.loads(args.selected.read_text())["ids"]
    if args.cache_kind:
        if not args.model_id:
            parser.error("--cache-kind requires --model-id")
        report = validate_cache(args.run, args.model_id, ids, args.cache_kind)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return
    payload = json.loads(args.run.read_text())
    if args.probe_kind:
        if not args.model_id or not args.expected_splits:
            parser.error("--probe-kind requires --model-id and --expected-splits")
        splits = json.loads(args.expected_splits.read_text())
        report = validate_probe(payload, args.model_id, ids, args.probe_kind, args.arm, args.readout, args.label_fraction, splits)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return
    visibility = json.loads(args.visibility.read_text()) if args.visibility else None
    report = validate_correspondence(payload, ids, visibility)
    if args.second or args.first_stage:
        second = json.loads(args.second.read_text()) if args.second else payload
        report["second"] = validate_correspondence(second, ids, visibility)
        if args.first_stage:
            report["paired"] = [validate_pairing(payload, second, f"{args.first_stage}:27", f"{args.second_stage or args.first_stage}:27")]
        else:
            report["paired"] = [validate_pairing(payload, second, stage) for stage in sorted(STAGES)]
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
