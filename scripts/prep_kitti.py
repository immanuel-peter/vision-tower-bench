"""Prepare KITTI's public selected validation set for the depth Transfer Probe."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_DIVISOR = 256.0
SOURCE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"


def sample_key(path: Path, kind: str) -> str:
    marker = f"_sync_{kind}_"
    if marker not in path.stem:
        raise ValueError(f"{path.name} does not contain {marker}")
    return path.stem.replace(marker, "_sync_", 1)


def paired_samples(root: Path) -> list[tuple[str, Path, Path]]:
    images = {sample_key(path, "image"): path for path in (root / "image").glob("*.png")}
    depths = {
        sample_key(path, "groundtruth_depth"): path
        for path in (root / "groundtruth_depth").glob("*.png")
    }
    if not images or images.keys() != depths.keys():
        missing_images = sorted(depths.keys() - images.keys())[:3]
        missing_depths = sorted(images.keys() - depths.keys())[:3]
        raise ValueError(
            f"unpaired KITTI selection: missing images {missing_images}, missing depths {missing_depths}"
        )
    return [(key, images[key], depths[key]) for key in sorted(images)]


def read_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        encoded = np.array(image, dtype=np.uint16)
    return encoded.astype(np.float32) / DEPTH_DIVISOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--cap", type=int, default=0, help="0 keeps all 1,000 public validation images")
    args = ap.parse_args()

    rows = paired_samples(args.raw)
    if args.cap:
        rows = rows[: args.cap]
    images = args.out / args.split
    images.mkdir(parents=True, exist_ok=True)

    targets: dict[str, np.ndarray] = {}
    source_ids: dict[str, str] = {}
    coverage: dict[str, float] = {}
    depth_max = 0.0
    for position, (source_id, image_path, depth_path) in enumerate(rows):
        image_id = f"{args.split}_{position:06d}"
        depth = read_depth(depth_path)
        shutil.copyfile(image_path, images / f"{image_id}.png")
        targets[image_id] = depth
        source_ids[image_id] = source_id
        coverage[image_id] = float((depth > 0).mean())
        depth_max = max(depth_max, float(depth.max()))
        if position % 100 == 0:
            print(f"{position}/{len(rows)}", flush=True)

    np.savez(args.out / f"{args.split}_targets.npz", **targets)
    values = list(coverage.values())
    manifest = {
        "source": "KITTI depth completion selected validation set",
        "source_url": SOURCE_URL,
        "split": args.split,
        "cap": args.cap,
        "images": len(rows),
        "metric_depth_divisor": DEPTH_DIVISOR,
        "max_depth_metres": round(depth_max, 3),
        "sparse_targets": True,
        "valid_pixel_rule": "finite metric depth greater than zero",
        "mean_valid_fraction": round(float(np.mean(values)), 6),
        "min_valid_fraction": round(float(np.min(values)), 6),
        "max_valid_fraction": round(float(np.max(values)), 6),
        "scenes": {image_id: "driving" for image_id in targets},
        "source_ids": source_ids,
    }
    (args.out / f"{args.split}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{len(rows)} samples -> {images}, depth reaches {depth_max:.3f} m, "
        f"mean valid fraction {np.mean(values):.4f}"
    )


if __name__ == "__main__":
    main()
