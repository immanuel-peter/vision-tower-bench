import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import rotate, uniform_filter1d


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


def condition_details(manifest: dict) -> dict[str, dict]:
    details = {"identity": {"factor": "identity", "level": 0, "value": 0}}
    names = {
        "scale": "scale",
        "occlusion": "occlusion",
        "motion_blur_radius_pixels": "motion_blur",
    }
    for source_name, factor in names.items():
        if source_name not in manifest["factors"]:
            continue
        for level, value in enumerate(manifest["factors"][source_name][1:], 1):
            details[f"{factor}_l{level}"] = {
                "factor": factor,
                "level": level,
                "value": value,
            }
    return details
