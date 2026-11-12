"""Run one image through a published MoonViT K3.1 Pro snapshot and its Projector.

Run from this downloaded bundle, or pass --model-dir for another pinned snapshot.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModel


import sys
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from projector import Projector


def square_image(image, resolution):
    width, height = image.size
    size = (resolution, int(resolution * height / width)) if width < height else (
        int(resolution * width / height), resolution
    )
    image = image.resize(size, Image.Resampling.BICUBIC)
    left, top = round((size[0] - resolution) / 2), round((size[1] - resolution) / 2)
    return image.crop((left, top, left + resolution, top + resolution))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=448)
    args = parser.parse_args()
    root = args.model_dir.absolute()
    if args.resolution % 28:
        parser.error("resolution must be divisible by 28 for 2x2 patch merging")
    for name in ("config.json", "model.safetensors", "projector_config.json", "projector.safetensors"):
        if not (root / name).is_file():
            parser.error(f"missing snapshot file: {root / name}")
    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(root, trust_remote_code=True, local_files_only=True, dtype=dtype)
    model = model.to(args.device).eval()
    processor = AutoImageProcessor.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    projector_config = json.loads((root / "projector_config.json").read_text())
    weights = load_file(root / "projector.safetensors")
    if set(weights) != {"proj.0.weight", "proj.2.weight", "post_norm.weight"}:
        raise ValueError("expected the three published Projector tensors")
    if any(value.dtype != dtype for value in weights.values()):
        raise ValueError("published Projector tensors must remain BF16")
    projector = Projector(projector_config).to(dtype=dtype)
    projector.load_state_dict(weights, strict=True)
    projector = projector.to(args.device).eval()
    with Image.open(args.image) as original:
        original_size = list(original.size)
        image = square_image(ImageOps.exif_transpose(original).convert("RGB"), args.resolution)
    inputs = processor(images=[image], return_tensors="pt")
    grid = inputs["grid_thws"]
    expected_grid = [1, args.resolution // model.config.patch_size, args.resolution // model.config.patch_size]
    if grid.tolist() != [expected_grid]:
        raise ValueError(f"unexpected single-image patch grid: {grid.tolist()}")
    inputs["pixel_values"] = inputs["pixel_values"].to(device=args.device, dtype=dtype)
    captured = {}

    def capture_final_tower(module, arguments, output):
        captured["tower"] = output.detach().unsqueeze(0)

    handle = model.encoder.register_forward_hook(capture_final_tower)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            packed = model(**inputs)
            if len(packed) != 1:
                raise ValueError("this example requires one independent image")
            merged = packed[0].reshape(1, -1, projector_config["input_size"])
            stages = captured | {"merged": merged, "projected": projector(merged)}
            tokens = expected_grid[1] * expected_grid[2]
            expected = {"tower": (1, tokens, 1024), "merged": (1, tokens // 4, 4096),
                        "projected": (1, tokens // 4, 7168)}
            for name, value in stages.items():
                if tuple(value.shape) != expected[name] or value.dtype != dtype or str(value.device) != args.device:
                    raise ValueError(f"unexpected {name} output: {value.shape}, {value.dtype}, {value.device}")
                if not torch.isfinite(value).all():
                    raise ValueError(f"nonfinite {name} output")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    finally:
        handle.remove()
    print(json.dumps({
        "model_dir": str(root), "image": str(args.image.absolute()),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "image_original_size": original_size, "device": args.device, "dtype": str(dtype),
        "batch_size": 1, "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "preprocessing": {"resize_short_edge": args.resolution, "interpolation": "PIL bicubic",
                          "center_crop": list(image.size), "processor": type(processor).__name__,
                          "image_mean": processor.image_mean, "image_std": processor.image_std,
                          "grid_thws": grid.tolist()},
        "stages": {name: {"shape": list(value.shape), "device": str(value.device), "dtype": str(value.dtype)}
                   for name, value in stages.items()},
        "forward_seconds": time.perf_counter() - started,
        "scope": "functional inference; does not establish official production TORCH_SDPA BF16 parity",
    }, indent=2))


if __name__ == "__main__":
    main()
