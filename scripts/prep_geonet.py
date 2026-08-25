"""Convert GeoNet NYU samples into images and targets the suite can read.

Each sample is a MATLAB file holding a mean-subtracted image padded to 481 by 641, a
depth map, precomputed surface normals, and a validity mask. This writes PNGs for
`vtb.extract` and one npz of targets keyed by image id for `vtb.geometry_run`.

The training set is capped (ADR-0011). Drawing the cap with a fixed seed keeps the same
images across every model, Stage, and depth point.
"""

import argparse
import io
import json
import zipfile
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


class Archives:
    """Reads GeoNet samples straight out of the zip archives.

    Extracting everything would need about 300 GB to keep 4,000 samples (ADR-0011).
    scipy reads a file object, so the archives stay closed and only the zips take disk.
    """

    def __init__(self, paths: list[Path]):
        self.zips = {path: zipfile.ZipFile(path) for path in paths}
        self.source = {
            Path(name).name: (path, name)
            for path, handle in self.zips.items()
            for name in handle.namelist()
            if name.endswith(".mat")
        }

    def names(self) -> list[str]:
        """Every sample across both archives, sorted, with Probe3D's two drops applied."""
        names = sorted(self.source)
        for index in BAD_INDICES:
            if index < len(names):
                del names[index]
        return names

    def open(self, name: str) -> bytes:
        path, member = self.source[name]
        with self.zips[path].open(member) as handle:
            return handle.read()

    def close(self) -> None:
        for handle in self.zips.values():
            handle.close()


def read(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = scipy.io.loadmat(io.BytesIO(raw))
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
    ap.add_argument("--zips", type=Path, nargs="+", required=True, help="data1.zip and data2.zip")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=("train", "valid"))
    ap.add_argument("--cap", type=int, default=4000, help="0 keeps every sample (ADR-0011)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    archives = Archives(args.zips)
    names = archives.names()
    keep = [n for i, n in enumerate(names) if (i % VALID_STRIDE == 0) == (args.split == "valid")]
    if args.cap and args.cap < len(keep):
        chosen = np.random.default_rng(args.seed).choice(len(keep), args.cap, replace=False)
        keep = [keep[i] for i in sorted(chosen)]

    images = args.out / args.split
    images.mkdir(parents=True, exist_ok=True)
    targets: dict[str, np.ndarray] = {}
    for position, name in enumerate(keep):
        image, depth, normal, mask = read(archives.open(name))
        image_id = f"{args.split}_{position:06d}"
        Image.fromarray(image).save(images / f"{image_id}.png")
        targets[image_id] = depth
        targets[f"{image_id}_normal"] = normal
        targets[f"{image_id}_valid"] = mask
        if position % 500 == 0:
            print(f"{position}/{len(keep)}", flush=True)

    archives.close()
    np.savez(args.out / f"{args.split}_targets.npz", **targets)
    (args.out / f"{args.split}_manifest.json").write_text(
        json.dumps({"source": [str(z) for z in args.zips], "split": args.split, "cap": args.cap,
                    "seed": args.seed, "images": len(keep), "files": keep}, indent=2) + "\n"
    )
    print(f"{len(keep)} samples -> {images}")


if __name__ == "__main__":
    main()
