"""Convert DIODE samples into images and targets the suite can read.

DIODE ships a PNG plus three npy arrays per sample: metric depth, a depth validity mask,
and surface normals. Normals arrive in a separate archive and cover only part of each
image, so validity comes from their magnitude rather than from a shipped mask.

Depth here is metric and reaches 230 m outdoors, against 10 m indoors on NYU. The depth
head bins over a fixed range, so the range is written into the manifest and passed to the
runner rather than assumed.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Below this magnitude a normal vector carries no direction, which is how DIODE marks
# pixels it could not annotate. Roughly half of an outdoor image is unannotated.
NORMAL_EPSILON = 1e-6


def samples(root: Path) -> dict[str, dict[str, Path]]:
    """Group the four files of each sample by its shared stem."""
    found: dict[str, dict[str, Path]] = {}
    for path in root.rglob("*.npy"):
        name = path.name
        for suffix, key in (("_depth_mask.npy", "mask"), ("_depth.npy", "depth"), ("_normal.npy", "normal")):
            if name.endswith(suffix):
                found.setdefault(name[: -len(suffix)], {})[key] = path
                break
    for path in root.rglob("*.png"):
        found.setdefault(path.stem, {})["image"] = path
    # Normals ship in their own archive, so a depth-only run is a valid intermediate state.
    return {stem: parts for stem, parts in found.items() if {"image", "depth", "mask"} <= parts.keys()}


def read(parts: dict[str, Path]):
    depth = np.load(parts["depth"]).squeeze(-1).astype(np.float32)
    mask = np.load(parts["mask"]).astype(np.float32)
    depth = depth * (mask > 0)
    if "normal" not in parts:
        return depth, None, None

    normal = np.load(parts["normal"]).astype(np.float32)
    magnitude = np.linalg.norm(normal, axis=-1)
    normal_valid = (magnitude > NORMAL_EPSILON).astype(np.uint8)
    return depth, normal.transpose(2, 0, 1), normal_valid


def copy_image(source: Path, destination: Path) -> None:
    """Copy the PNG rather than re-encode it. DIODE already ships RGB PNGs, and
    re-encoding 771 of them is the only part of this step that works the CPU hard."""
    with Image.open(source) as probe:
        if probe.mode == "RGB":
            shutil.copyfile(source, destination)
            return
        probe.convert("RGB").save(destination)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="directory holding the extracted tarballs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--scene", default="all", choices=("all", "indoors", "outdoor"))
    ap.add_argument("--cap", type=int, default=0, help="0 keeps every sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    found = samples(args.raw)
    stems = sorted(s for s in found if args.scene == "all" or f"_{args.scene}_" in s)
    if args.cap and args.cap < len(stems):
        chosen = np.random.default_rng(args.seed).choice(len(stems), args.cap, replace=False)
        stems = [stems[i] for i in sorted(chosen)]

    images = args.out / args.split
    images.mkdir(parents=True, exist_ok=True)
    targets: dict[str, np.ndarray] = {}
    scenes: dict[str, str] = {}
    depth_max = 0.0
    for position, stem in enumerate(stems):
        depth, normal, valid = read(found[stem])
        image_id = f"{args.split}_{position:06d}"
        copy_image(found[stem]["image"], images / f"{image_id}.png")
        targets[image_id] = depth
        if normal is not None:
            targets[f"{image_id}_normal"] = normal.astype(np.float16)
            targets[f"{image_id}_valid"] = valid
        # DIODE encodes the scene type in the file name, which gives the writeup an
        # indoor against outdoor split for free.
        scenes[image_id] = "indoors" if "_indoors_" in stem else "outdoor"
        depth_max = max(depth_max, float(depth.max()))
        if position % 100 == 0:
            print(f"{position}/{len(stems)}", flush=True)

    np.savez(args.out / f"{args.split}_targets.npz", **targets)
    (args.out / f"{args.split}_manifest.json").write_text(
        json.dumps({"source": "DIODE", "split": args.split, "scene": args.scene,
                    "cap": args.cap, "seed": args.seed, "images": len(stems),
                    "max_depth_metres": round(depth_max, 2),
                    "has_normals": any(k.endswith("_normal") for k in targets),
                    "scenes": scenes}, indent=2) + "\n"
    )
    print(f"{len(stems)} samples -> {images}, depth reaches {depth_max:.1f} m")


if __name__ == "__main__":
    main()
