"""Convert GeoNet NYU samples into images and targets the suite can read.

Each sample is a MATLAB file holding a mean-subtracted image padded to 481 by 641, a
depth map, precomputed surface normals, and a validity mask. This writes PNGs for
`vtb.extract` and one npz of targets keyed by image id for `vtb.geometry_run`.

The training set is capped (ADR-0011). Drawing the cap with a fixed seed keeps the same
images across every model, Stage, and depth point.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

# GeoNet stores doubled mean-centered images.
CHANNEL_MEANS = (122.175, 116.169, 103.508)
MAX_DEPTH = 10.0
# Match Probe3D's two dropped samples.
BAD_INDICES = (21181, 6919)
VALID_STRIDE = 20


def instances(root: Path) -> list[str]:
    names = sorted(p.name for p in root.iterdir() if p.suffix == ".mat")
    for index in BAD_INDICES:
        if index < len(names):
            del names[index]
    return names


def read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = scipy.io.loadmat(str(path))
    image = sample["img"][:480, :640].astype(np.float32)
    for channel, mean in enumerate(CHANNEL_MEANS):
        image[:, :, channel] += 2 * mean
    depth = sample["depth"][:480, :640].astype(np.float32)
    depth[depth > MAX_DEPTH] = 0.0
    normal = sample["norm"][:480, :640].transpose(2, 0, 1).astype(np.float16)
    mask = sample["mask"][:480, :640].astype(np.uint8) if "mask" in sample else (depth > 0).astype(np.uint8)
    return image.clip(0, 255).astype(np.uint8), depth, normal, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="directory of GeoNet .mat files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=("train", "valid"))
    ap.add_argument("--cap", type=int, default=4000, help="0 keeps every sample (ADR-0011)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = instances(args.raw)
    keep = [n for i, n in enumerate(names) if (i % VALID_STRIDE == 0) == (args.split == "valid")]
    if args.cap and args.cap < len(keep):
        chosen = np.random.default_rng(args.seed).choice(len(keep), args.cap, replace=False)
        keep = [keep[i] for i in sorted(chosen)]

    images = args.out / args.split
    images.mkdir(parents=True, exist_ok=True)
    targets: dict[str, np.ndarray] = {}
    for position, name in enumerate(keep):
        image, depth, normal, mask = read(args.raw / name)
        image_id = f"{args.split}_{position:06d}"
        Image.fromarray(image).save(images / f"{image_id}.png")
        targets[image_id] = depth
        targets[f"{image_id}_normal"] = normal
        targets[f"{image_id}_valid"] = mask
        if position % 500 == 0:
            print(f"{position}/{len(keep)}", flush=True)

    np.savez(args.out / f"{args.split}_targets.npz", **targets)
    (args.out / f"{args.split}_manifest.json").write_text(
        json.dumps({"source": str(args.raw), "split": args.split, "cap": args.cap,
                    "seed": args.seed, "images": len(keep), "files": keep}, indent=2) + "\n"
    )
    print(f"{len(keep)} samples -> {images}")


if __name__ == "__main__":
    main()
