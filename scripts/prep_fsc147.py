"""Prepare FSC-147 count-v1 targets: centre-crop PNGs, densities, exemplar boxes, and a letterbox val set."""

import argparse
import hashlib
import io
import json
import os
import subprocess
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from vtb import counting

SPLITS = ("train", "val", "test")
EXPECTED_SPLITS = {"train": 3659, "val": 1286, "test": 1190}
EXPECTED_CLASSES = 147
RELEASE_HEIGHT = 384
EXEMPLARS = 3
RESOLUTIONS = (448, 512)
ANNOTATIONS = "annotation_FSC147_384.json"
SPLIT_FILE = "Train_Test_Val_FSC_147.json"
CLASS_FILE = "ImageClasses_FSC147.txt"
TARGETS = "fsc147_targets.npz"
MANIFEST = "fsc147_manifest.json"
LETTERBOX_TARGETS = "letterbox_val_targets.npz"
LETTERBOX_MANIFEST = "letterbox_val_manifest.json"
COMPLETE = "fsc147_prep_complete.json"
REPO_COMMIT = "00ea1888c3a7c495ae06db0bddc3b90b7db8d52f"
SOURCE = {
    "release": "https://github.com/cvlab-stonybrook/LearningToCountEverything",
    "release_commit": REPO_COMMIT,
    "annotation_url": "https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/"
                      f"{REPO_COMMIT}/data/",
    "images_url": "https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing",
    "hf_mirror": "isentropic/FSC147",
    "hf_mirror_revision": "3e420cb6537e803dd6d4516623ce82a79c0317b8",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` via a temp file; untouched when the bytes already match."""
    if path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """An np.load-compatible npz with fixed zip timestamps, so equal arrays give equal bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for key, array in arrays.items():
            entry = io.BytesIO()
            np.lib.format.write_array(entry, np.ascontiguousarray(array), allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)), entry.getvalue())
    return buffer.getvalue()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1) + "\n").encode()


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_id(file_name: str) -> str:
    return f"fsc147_{Path(file_name).stem}"


def corners_to_box(corners) -> np.ndarray:
    """Four (x, y) corners to (x0, y0, x1, y1)."""
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(corners) != 4:
        raise ValueError(f"expected 4 corners, got {len(corners)}")
    return np.concatenate([corners.min(axis=0), corners.max(axis=0)])


def load_release(raw: Path) -> tuple[dict, dict, dict]:
    annotations = json.loads((raw / ANNOTATIONS).read_text())
    splits = json.loads((raw / SPLIT_FILE).read_text())
    classes = {}
    for line in (raw / CLASS_FILE).read_text().splitlines():
        if line.strip():
            name, label = line.split("\t", 1)
            if name in classes and classes[name] != label.strip():
                raise ValueError(f"{name} has two classes")
            classes[name] = label.strip()
    return annotations, splits, classes


def check_release(annotations: dict, splits: dict, classes: dict, expected: dict | None = EXPECTED_SPLITS,
                  expected_classes: int | None = EXPECTED_CLASSES) -> dict:
    """Release-level checks. ``problems`` lists every violation; prep refuses to run on any."""
    problems = []
    members = {split: list(splits.get(split, [])) for split in SPLITS}
    every = [name for split in SPLITS for name in members[split]]
    seen = Counter(every)
    problems += [f"{name} appears {n} times across splits" for name, n in sorted(seen.items()) if n > 1]
    for kind, key in (("stem", lambda n: Path(n).stem), ("image_id", image_id)):
        owners: dict[str, set] = {}
        for name in seen:
            owners.setdefault(key(name), set()).add(name)
        problems += [f"{kind} collision {k}: {sorted(v)}" for k, v in sorted(owners.items()) if len(v) > 1]
    split_classes = {}
    boxes_per_image = Counter()
    for split in SPLITS:
        labels = set()
        for name in members[split]:
            entry = annotations.get(name)
            if entry is None:
                problems.append(f"{name} ({split}) has no annotation")
                continue
            if name not in classes:
                problems.append(f"{name} ({split}) has no class")
            else:
                labels.add(classes[name])
            if not entry.get("points"):
                problems.append(f"{name} has no points")
            boxes = entry.get("box_examples_coordinates", [])
            boxes_per_image[len(boxes)] += 1
            if len(boxes) < EXEMPLARS:
                problems.append(f"{name} has {len(boxes)} boxes")
            for corners in boxes:
                if len(corners) != 4:
                    problems.append(f"{name} has a box with {len(corners)} corners")
        split_classes[split] = labels
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            shared = split_classes[a] & split_classes[b]
            if shared:
                problems.append(f"classes shared by {a} and {b}: {sorted(shared)}")
    all_classes = set().union(*split_classes.values())
    sizes = {split: len(members[split]) for split in SPLITS}
    if expected is not None:
        problems += [f"{s} has {sizes[s]} images, the release states {n}" for s, n in expected.items() if sizes[s] != n]
    if expected_classes is not None and len(all_classes) != expected_classes:
        problems.append(f"{len(all_classes)} classes, the release states {expected_classes}")
    return {
        "problems": problems,
        "split_sizes": sizes,
        "images": len(seen),
        "classes": len(all_classes),
        "classes_per_split": {s: len(split_classes[s]) for s in SPLITS},
        "classes_disjoint": not any("classes shared" in p for p in problems),
        "boxes_per_image": {str(k): v for k, v in sorted(boxes_per_image.items())},
        "annotations_outside_splits": sorted(set(annotations) - set(seen), key=lambda n: (len(n), n)),
        "other_split_keys": {k: len(v) for k, v in splits.items() if k not in SPLITS},
    }


def exemplar_boxes(frame: counting.SquareFrame, corners: list) -> tuple[list, list, list, list]:
    """Release boxes 0..2 on the frame: clipped unit boxes, kept fractions, validity, reasons."""
    boxes, kept, valid, reasons = [], [], [], []
    for index, unit in enumerate(frame.boxes(np.stack([corners_to_box(c) for c in corners[:EXEMPLARS]]))):
        try:
            clipped, fraction = counting.clip_box(unit)
            reason = f"box {index} keeps {fraction:.3f} of its area (< {counting.MIN_BOX_AREA_KEPT})"
        except ValueError:
            clipped, fraction, reason = unit, 0.0, f"box {index} is degenerate"
        ok = bool(fraction >= counting.MIN_BOX_AREA_KEPT)
        if not ok:
            reasons.append(reason)
        boxes.append([float(v) for v in clipped])
        kept.append(float(fraction))
        valid.append(ok)
    return boxes, kept, valid, reasons


def frame_targets(frame: counting.SquareFrame, dots: np.ndarray, corners: list) -> dict:
    unit = frame.points(dots)
    keep = counting.inside(unit)
    boxes, kept, valid, reasons = exemplar_boxes(frame, corners)
    count = int(keep.sum())
    return {"frame": frame.as_dict(), "source_count": len(dots), "count": count, "bin": counting.count_bin(count),
            "boxes": boxes, "box_kept": kept, "box_valid": valid, "exemplar_eligible": all(valid),
            "density": counting.density_map(unit[keep]), "box_reasons": reasons}


def prepare_one(task: dict) -> dict:
    """Decode one release image, write its crop (and letterbox) PNG, and build its targets."""
    data = Path(task["path"]).read_bytes()
    with Image.open(io.BytesIO(data)) as decoded:
        mode = decoded.mode
        image = decoded.convert("RGB")
    width, height = image.size
    entry = task["annotation"]
    release_dots = np.asarray(entry["points"], dtype=np.float64).reshape(-1, 2)
    dots, edge_dots = counting.onto_image(release_dots, width, height)
    corners = entry["box_examples_coordinates"]
    out, iid = Path(task["out"]), task["image_id"]

    crop = counting.centre_crop_frame(width, height)
    targets = frame_targets(crop, dots, corners)
    png = png_bytes(image.crop((crop.left, crop.top, crop.left + crop.side, crop.top + crop.side)))
    write_atomic(out / "images" / f"{iid}.png", png)

    notes = []
    if mode != "RGB":
        notes.append(f"source mode {mode} converted to RGB")
    if len(corners) > EXEMPLARS:
        notes.append(f"release has {len(corners)} boxes; boxes 0-2 used")
    source_box = np.stack([corners_to_box(c) for c in corners[:EXEMPLARS]])
    if edge_dots:
        notes.append(f"{edge_dots} release dots on the image edge clamped onto it")
    outside_dots = int((~counting.inside(dots / (width, height))).sum())
    if outside_dots:
        notes.append(f"{outside_dots} release dots lie outside the source image")
    if (source_box[:, :2] < 0).any() or (source_box[:, 2] > width).any() or (source_box[:, 3] > height).any():
        notes.append("a release box extends past the source image")
    # The 384 release resized each original by ratio_w, ratio_h.
    expected_width = entry["W"] * entry["ratio_w"]
    expected_height = entry["H"] * entry["ratio_h"]
    record = {
        "source_file": task["file"], "source_sha256": sha256(data), "png_sha256": sha256(png),
        "class": task["class"], "split": task["split"], "width": width, "height": height,
        **{k: v for k, v in targets.items() if k not in ("density", "box_reasons")},
        "release_boxes": len(corners),
        "exclusions": [f"exemplar arm: {r}" for r in targets["box_reasons"]],
        "notes": notes,
    }
    result = {"image_id": iid, "record": record, "density": targets["density"],
              "size_vs_annotation": [width - expected_width, height - expected_height]}

    if task["letterbox"]:
        frame = counting.letterbox_frame(width, height)
        canvas = Image.new("RGB", (frame.side, frame.side), counting.LETTERBOX_FILL)
        canvas.paste(image, (-frame.left, -frame.top))
        lb_png = png_bytes(canvas)
        write_atomic(out / "letterbox_val" / f"{iid}.png", lb_png)
        lb = frame_targets(frame, dots, corners)
        result["letterbox"] = {
            "source_file": task["file"], "png_sha256": sha256(lb_png), "class": task["class"],
            "split": task["split"], "width": width, "height": height, "valid": list(frame.valid),
            **{k: v for k, v in lb.items() if k not in ("density", "box_reasons")},
            "dropped_dots": lb["source_count"] - lb["count"],
        }
        result["letterbox_density"] = lb["density"]
    return result


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def summarize(records: dict, ids: list[str]) -> dict:
    rows = [records[i] for i in ids]
    bins = Counter(r["bin"] for r in rows)
    return {
        "images": len(rows),
        "eligible": sum(r["exemplar_eligible"] for r in rows),
        "zero_count": sum(r["count"] == 0 for r in rows),
        "zero_count_ids": [i for i in ids if records[i]["count"] == 0],
        "dropping_dots": sum(r["count"] < r["source_count"] for r in rows),
        "source_dots": sum(r["source_count"] for r in rows),
        "retained_dots": sum(r["count"] for r in rows),
        "bins": {b: bins.get(b, 0) for b in counting.BINS},
        "more_than_3_boxes": sum(r["release_boxes"] > EXEMPLARS for r in rows),
    }


def side_summary(sides: list[int]) -> dict:
    sides_arr = np.asarray(sides, dtype=np.float64)
    quantiles = {f"p{q}": float(np.percentile(sides_arr, q)) for q in (0, 5, 50, 95, 100)}
    upsample = {
        str(r): {k: round(r / v, 4) for k, v in (("max", quantiles["p0"]), ("median", quantiles["p50"]),
                                                  ("min", quantiles["p100"]))}
        | {"images_upsampled": int((sides_arr < r).sum())}
        for r in RESOLUTIONS
    }
    return {"side": quantiles, "side_counts": {str(k): v for k, v in sorted(Counter(sides).items())},
            "upsample_factor": upsample}


def reference_check(records: dict, reference: Path) -> dict:
    """Compare source image bytes with another copy of the release (a directory or zip of the jpgs)."""
    wanted = {r["source_file"]: r["source_sha256"] for r in records.values()}
    found = {}
    if reference.is_dir():
        for path in reference.rglob("*.jpg"):
            if path.name in wanted:
                found[path.name] = file_sha256(path)
    else:
        with zipfile.ZipFile(reference) as archive:
            for name in archive.namelist():
                base = Path(name).name
                if base in wanted and name.endswith(".jpg") and "__MACOSX" not in name:
                    found[base] = sha256(archive.read(name))
    missing = sorted(set(wanted) - set(found))
    differ = sorted(n for n in found if found[n] != wanted[n])
    return {"reference": str(reference), "compared": len(found), "missing": missing, "differ": differ,
            "identical": not missing and not differ}


def density_shape_check(records: dict, density_zip: Path) -> dict:
    """Compare image sizes with the authors' precomputed density maps, reading only the npy headers."""
    by_stem = {Path(r["source_file"]).stem: (r["height"], r["width"]) for r in records.values()}
    mismatched, found = [], 0
    with zipfile.ZipFile(density_zip) as archive:
        for name in archive.namelist():
            stem = Path(name).stem
            if not name.endswith(".npy") or stem not in by_stem or "__MACOSX" in name:
                continue
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, _ = read(handle)
            found += 1
            if tuple(shape) != by_stem[stem]:
                mismatched.append(f"{stem}: density {tuple(shape)}, image {by_stem[stem]}")
    return {"density_zip": str(density_zip), "density_zip_sha256": file_sha256(density_zip), "compared": found,
            "missing": len(by_stem) - found, "mismatched": mismatched}


def validate(out: Path) -> dict:
    """Re-read a prepared directory and check IDs, densities, PNGs and the completion record."""
    complete = json.loads((out / COMPLETE).read_text())
    for name, digest in complete["sha256"].items():
        if file_sha256(out / name) != digest:
            raise ValueError(f"{name} does not match its completion record")
    for manifest_name, targets_name, folder, ids_key in (
        (MANIFEST, TARGETS, "images", "splits"), (LETTERBOX_MANIFEST, LETTERBOX_TARGETS, "letterbox_val", "ids"),
    ):
        manifest = json.loads((out / manifest_name).read_text())
        records = manifest["images"]
        listed = manifest[ids_key]
        ids = [i for s in SPLITS for i in listed[s]] if ids_key == "splits" else listed
        if len(set(ids)) != len(ids) or set(ids) != set(records):
            raise ValueError(f"{manifest_name}: split ids and image records disagree")
        with np.load(out / targets_name) as targets:
            if set(targets.files) != set(ids):
                raise ValueError(f"{targets_name} keys differ from {manifest_name}")
            for iid in ids:
                density = targets[iid]
                record = records[iid]
                if density.shape != (counting.TARGET_GRID,) * 2 or density.dtype != np.float32:
                    raise ValueError(f"{iid}: density {density.shape} {density.dtype}")
                if abs(float(density.sum(dtype=np.float64)) - record["count"]) > 1e-3 * max(1, record["count"]):
                    raise ValueError(f"{iid}: density sums to {density.sum()}, count {record['count']}")
                if record["bin"] != counting.count_bin(record["count"]):
                    raise ValueError(f"{iid}: bin")
                if record["exemplar_eligible"] != all(record["box_valid"]) or len(record["boxes"]) != EXEMPLARS:
                    raise ValueError(f"{iid}: eligibility")
        pngs = {p.stem: p for p in (out / folder).glob("*.png")}
        if set(pngs) != set(ids):
            raise ValueError(f"{folder} holds {len(pngs)} PNGs for {len(ids)} ids")
        for iid in ids:
            side = records[iid]["frame"]["side"]
            if file_sha256(pngs[iid]) != records[iid]["png_sha256"]:
                raise ValueError(f"{iid}: PNG bytes changed")
            with Image.open(pngs[iid]) as image:
                if image.size != (side, side) or image.mode != "RGB":
                    raise ValueError(f"{iid}: PNG is {image.size} {image.mode}")
    return complete


def code_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, help="directory holding the three release annotation files")
    ap.add_argument("--images", type=Path, help="directory holding the release jpgs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--validate", action="store_true", help="only re-check an existing --out")
    ap.add_argument("--ids", default="", help="comma-separated release stems for a fixture; empty keeps all")
    ap.add_argument("--archive", type=Path, help="the Drive zip, recorded by sha256 when present")
    ap.add_argument("--transport", default="drive", help="how the images arrived, e.g. drive or hf-mirror")
    ap.add_argument("--reference-images", type=Path, help="official jpg directory or zip to compare bytes with")
    ap.add_argument("--reference-density", type=Path, help="official density-map zip to compare sizes with")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    start = time.time()
    if args.validate:
        print(json.dumps(validate(args.out)["sha256"], indent=1))
        return
    if args.raw is None or args.images is None:
        ap.error("--raw and --images are required unless --validate")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / COMPLETE).unlink(missing_ok=True)
    annotations, splits, classes = load_release(args.raw)
    release = check_release(annotations, splits, classes, EXPECTED_SPLITS, EXPECTED_CLASSES)
    if release["problems"]:
        raise SystemExit("release check failed:\n  " + "\n  ".join(release["problems"]))

    wanted = {s.strip().removesuffix(".jpg") for s in args.ids.split(",") if s.strip()}
    tasks = []
    for split in SPLITS:
        for name in splits[split]:
            if wanted and Path(name).stem not in wanted:
                continue
            tasks.append({"image_id": image_id(name), "file": name, "path": str(args.images / name),
                          "annotation": annotations[name], "class": classes[name], "split": split,
                          "out": str(args.out), "letterbox": split == "val"})
    if wanted and len(tasks) != len(wanted):
        raise SystemExit(f"--ids names {len(wanted)} stems, {len(tasks)} are in the splits")
    if len({t["image_id"] for t in tasks}) != len(tasks):
        raise SystemExit("duplicate image ids")
    for sub in ("images", "letterbox_val"):
        (args.out / sub).mkdir(exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        results = []
        for position, result in enumerate(pool.map(prepare_one, tasks, chunksize=16)):
            results.append(result)
            if position % 500 == 0:
                print(f"{position}/{len(tasks)}", flush=True)

    # Drop PNGs a previous run left behind, since feature extraction globs the directory.
    for sub, keep in (("images", {r["image_id"] for r in results}),
                      ("letterbox_val", {r["image_id"] for r in results if "letterbox" in r})):
        for path in (args.out / sub).glob("*.png"):
            if path.stem not in keep:
                path.unlink()

    records = {r["image_id"]: r["record"] for r in results}
    split_ids = {s: [r["image_id"] for r in results if r["record"]["split"] == s] for s in SPLITS}
    off = [(i, d) for i, d in ((r["image_id"], r["size_vs_annotation"]) for r in results)
           if abs(d[0]) > 1 or abs(d[1]) > 1]
    heights = Counter(r["height"] for r in records.values())
    images_check = {
        "heights": {str(k): v for k, v in sorted(heights.items())},
        "not_release_height": [i for i, r in records.items() if r["height"] != RELEASE_HEIGHT],
        "size_off_annotation_ratio_by_more_than_1px": [f"{i}: dw {d[0]:+.2f}, dh {d[1]:+.2f}" for i, d in off],
        "non_rgb_sources": [i for i, r in records.items() if any("converted" in n for n in r["notes"])],
        "dots_outside_source": [i for i, r in records.items() if any("dots lie outside" in n for n in r["notes"])],
        "boxes_past_source": [i for i, r in records.items() if any("box extends" in n for n in r["notes"])],
    }
    if args.reference_images:
        images_check["reference_images"] = reference_check(records, args.reference_images)
    if args.reference_density:
        images_check["reference_density"] = density_shape_check(records, args.reference_density)

    source = dict(SOURCE, transport=args.transport,
                  annotation_sha256={f: file_sha256(args.raw / f) for f in (ANNOTATIONS, SPLIT_FILE, CLASS_FILE)},
                  archive=str(args.archive) if args.archive else None,
                  archive_sha256=file_sha256(args.archive) if args.archive and args.archive.exists() else None,
                  images_dir=str(args.images),
                  source_images_sha256=sha256("".join(f"{r['source_file']} {r['source_sha256']}\n"
                                                      for r in records.values()).encode()))
    exclusions = [{"image_id": i, "scope": "exemplar", "reason": reason}
                  for i, r in records.items() for reason in r["exclusions"]]
    counts = {s: summarize(records, split_ids[s]) for s in SPLITS}
    counts["all"] = summarize(records, list(records))
    counts["crop"] = side_summary([r["frame"]["side"] for r in records.values()])
    manifest = {
        "source": source, "protocol": counting.PROTOCOL, "target_grid": counting.TARGET_GRID,
        "sigma_cells": counting.SIGMA_CELLS, "truncate_sigmas": counting.TRUNCATE_SIGMAS,
        "min_box_area_kept": counting.MIN_BOX_AREA_KEPT, "exemplars": EXEMPLARS,
        "fixture": bool(wanted), "images": records, "splits": split_ids, "counts": counts,
        "exclusions": exclusions, "verification": {"release": release, "images": images_check},
    }
    targets = npz_bytes({r["image_id"]: r["density"] for r in results})
    manifest_data = json_bytes(manifest)
    write_atomic(args.out / TARGETS, targets)
    write_atomic(args.out / MANIFEST, manifest_data)

    lb_records = {r["image_id"]: r["letterbox"] for r in results if "letterbox" in r}
    lb_ids = list(lb_records)
    lb_manifest = {
        "source_manifest_sha256": sha256(manifest_data), "protocol": counting.PROTOCOL,
        "frame_kind": "letterbox", "fill": list(counting.LETTERBOX_FILL), "target_grid": counting.TARGET_GRID,
        "images": lb_records, "ids": lb_ids,
        "counts": {"images": len(lb_ids), "dots": sum(r["count"] for r in lb_records.values()),
                   "dropped_dots": sum(r["dropped_dots"] for r in lb_records.values()),
                   "eligible": sum(r["exemplar_eligible"] for r in lb_records.values()),
                   "bins": dict(Counter(r["bin"] for r in lb_records.values())),
                   "sides": side_summary([r["frame"]["side"] for r in lb_records.values()]) if lb_ids else {}},
    }
    lb_targets = npz_bytes({r["image_id"]: r["letterbox_density"] for r in results if "letterbox" in r})
    lb_manifest_data = json_bytes(lb_manifest)
    write_atomic(args.out / LETTERBOX_TARGETS, lb_targets)
    write_atomic(args.out / LETTERBOX_MANIFEST, lb_manifest_data)

    complete = {
        "protocol": counting.PROTOCOL, "images": len(records), "letterbox_images": len(lb_ids),
        "sha256": {TARGETS: sha256(targets), MANIFEST: sha256(manifest_data),
                   LETTERBOX_TARGETS: sha256(lb_targets), LETTERBOX_MANIFEST: sha256(lb_manifest_data)},
        "code_commit": code_commit(), "numpy": np.__version__, "pillow": PIL.__version__,
        "args": {k: str(v) for k, v in vars(args).items()}, "seconds": round(time.time() - start, 1),
    }
    write_atomic(args.out / COMPLETE, json_bytes(complete))
    try:
        validate(args.out)
    except ValueError:
        (args.out / COMPLETE).unlink()
        raise
    print(json.dumps({s: counts[s] | {"zero_count_ids": len(counts[s]["zero_count_ids"])} for s in (*SPLITS, "all")},
                     indent=1))
    print(f"{len(records)} images -> {args.out}, {len(lb_ids)} letterbox, {complete['seconds']} s")


if __name__ == "__main__":
    main()
