"""Create the fixed ImageNet-100 Perturbation Study image set and metadata."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import rotate, uniform_filter1d

from vtb.images import SUFFIXES

SCALE_LEVELS = (1.0, 1.15, 1.35, 1.65, 2.0)
OCCLUSION_LEVELS = (0.0, 0.1, 0.2, 0.35, 0.5)
BLUR_RADII = (0, 2, 4, 8, 12)


def centre_scale(image: Image.Image, apparent_scale: float) -> tuple[Image.Image, dict]:
    width, height = image.size
    crop_width = max(1, round(width / apparent_scale))
    crop_height = max(1, round(height / apparent_scale))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    box = (left, top, left + crop_width, top + crop_height)
    transformed = image.crop(box).resize((width, height), Image.Resampling.BICUBIC)
    return transformed, {
        "apparent_scale": apparent_scale,
        "crop_box_pixels": list(box),
        "crop_fraction": round((crop_width * crop_height) / (width * height), 6),
    }


def occlude(
    image: Image.Image, area_fraction: float, rng: np.random.Generator
) -> tuple[Image.Image, dict]:
    width, height = image.size
    rectangle_width = max(1, round(width * area_fraction**0.5))
    rectangle_height = max(1, round(height * area_fraction**0.5))
    left = int(rng.integers(0, width - rectangle_width + 1))
    top = int(rng.integers(0, height - rectangle_height + 1))
    box = (left, top, left + rectangle_width, top + rectangle_height)
    transformed = image.copy()
    ImageDraw.Draw(transformed).rectangle(
        (left, top, left + rectangle_width - 1, top + rectangle_height - 1),
        fill=(127, 127, 127),
    )
    return transformed, {
        "requested_area_fraction": area_fraction,
        "actual_area_fraction": round((rectangle_width * rectangle_height) / (width * height), 6),
        "rectangle_pixels": list(box),
        "fill_rgb": [127, 127, 127],
    }


def directional_blur(image: Image.Image, radius: int, angle_degrees: float) -> Image.Image:
    values = np.asarray(image, dtype=np.float32)
    aligned = rotate(values, -angle_degrees, axes=(1, 0), reshape=False, order=1, mode="reflect")
    blurred = uniform_filter1d(aligned, size=2 * radius + 1, axis=1, mode="nearest")
    restored = rotate(blurred, angle_degrees, axes=(1, 0), reshape=False, order=1, mode="reflect")
    return Image.fromarray(np.clip(restored, 0, 255).astype(np.uint8), "RGB")


def save(image: Image.Image, path: Path) -> None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=95)
    else:
        image.save(path)


def process_image(task: tuple[int, str, str, int, tuple[str, ...]]) -> list[dict]:
    position, source_text, out_text, seed, factors = task
    source, out = Path(source_text), Path(out_text)
    image_id = source.stem
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB").copy()
    rows = []
    for factor in factors:
        rows.append({
            "image_id": image_id,
            "source": str(source),
            "factor": factor,
            "level": 0,
            "condition": "identity",
            "parameters": {"identity": True},
        })

    if "scale" in factors:
        for level, value in enumerate(SCALE_LEVELS[1:], 1):
            transformed, parameters = centre_scale(image, value)
            condition = f"scale_l{level}"
            destination = out / condition / source.name
            save(transformed, destination)
            rows.append({"image_id": image_id, "source": str(source), "factor": "scale",
                         "level": level, "condition": condition, "parameters": parameters})

    if "occlusion" in factors:
        for level, value in enumerate(OCCLUSION_LEVELS[1:], 1):
            rng = np.random.default_rng(seed + position * 101 + level)
            transformed, parameters = occlude(image, value, rng)
            condition = f"occlusion_l{level}"
            destination = out / condition / source.name
            save(transformed, destination)
            rows.append({"image_id": image_id, "source": str(source), "factor": "occlusion",
                         "level": level, "condition": condition, "parameters": parameters})

    if "motion_blur" in factors:
        angle = float(np.random.default_rng(seed + position * 101 + 97).uniform(0, 180))
        for level, radius in enumerate(BLUR_RADII[1:], 1):
            transformed = directional_blur(image, radius, angle)
            condition = f"motion_blur_l{level}"
            destination = out / condition / source.name
            save(transformed, destination)
            rows.append({
                "image_id": image_id,
                "source": str(source),
                "factor": "motion_blur",
                "level": level,
                "condition": condition,
                "parameters": {"radius_pixels": radius, "kernel_length": 2 * radius + 1,
                               "angle_degrees": round(angle, 6)},
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--factors", nargs="+", choices=("scale", "occlusion", "motion_blur"),
        default=["scale", "occlusion", "motion_blur"],
    )
    args = ap.parse_args()

    sources = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in SUFFIXES)
    sources = sources[: args.limit]
    if len(sources) != args.limit:
        raise SystemExit(f"found {len(sources)} images, expected {args.limit}")
    conditions = []
    if "scale" in args.factors:
        conditions.extend(f"scale_l{i}" for i in range(1, len(SCALE_LEVELS)))
    if "occlusion" in args.factors:
        conditions.extend(f"occlusion_l{i}" for i in range(1, len(OCCLUSION_LEVELS)))
    if "motion_blur" in args.factors:
        conditions.extend(f"motion_blur_l{i}" for i in range(1, len(BLUR_RADII)))
    for condition in conditions:
        (args.out / condition).mkdir(parents=True, exist_ok=True)

    factors = tuple(dict.fromkeys(args.factors))
    tasks = [(i, str(path), str(args.out), args.seed, factors) for i, path in enumerate(sources)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for position, result in enumerate(pool.map(process_image, tasks)):
            rows.extend(result)
            if position % 100 == 0:
                print(f"{position}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (row["factor"], row["level"], row["image_id"]))

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "source": "ImageNet-100 validation",
        "source_root": str(args.images),
        "images": len(sources),
        "seed": args.seed,
        "factors": {
            **({"scale": list(SCALE_LEVELS)} if "scale" in factors else {}),
            **({"occlusion": list(OCCLUSION_LEVELS)} if "occlusion" in factors else {}),
            **({"motion_blur_radius_pixels": list(BLUR_RADII)}
               if "motion_blur" in factors else {}),
        },
        "identity_root": str(args.images),
        "condition_root": str(args.out),
        "conditions": ["identity", *conditions],
        "metadata_rows": len(rows),
    }
    args.metadata.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} transform rows for {len(sources)} images")


if __name__ == "__main__":
    main()
