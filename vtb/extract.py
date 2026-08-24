import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vtb.adapters.dinov2 import DINOv2Adapter
from vtb.images import ImageFolder

ADAPTERS = {"dinov2": DINOv2Adapter}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frozen Tower features and measure cache growth.")
    ap.add_argument("--model", default="dinov2", choices=sorted(ADAPTERS))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("cache"))
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    adapter = ADAPTERS[args.model](resolution=args.resolution, device=args.device)
    dataset = ImageFolder(args.images, args.resolution, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, collate_fn=list_collate)

    run_dir = args.out / f"{args.model}_{args.resolution}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(dataset)} images, {adapter.num_layers} layers, depth points {adapter.depth_points()}")

    written = images = 0
    start = time.perf_counter()
    for shard, (pixel_values, image_ids) in enumerate(loader):
        for batch in adapter.extract(pixel_values, image_ids):
            path = run_dir / f"L{batch.layer_index:02d}_{shard:05d}.safetensors"
            batch.save(path)
            written += batch.nbytes
        images += len(image_ids)
        elapsed = time.perf_counter() - start
        print(f"{images}/{len(dataset)}  {images / elapsed:5.1f} img/s  {written / 1e9:6.2f} GB", flush=True)

    report(run_dir, adapter, images, written, time.perf_counter() - start)


def list_collate(samples):
    pixel_values, image_ids = zip(*samples)
    return torch.stack(pixel_values), list(image_ids)


def report(run_dir: Path, adapter, images: int, written: int, elapsed: float) -> None:
    n_points = len(adapter.depth_points())
    per_image = written / images
    per_point = per_image / n_points
    on_disk = sum(p.stat().st_size for p in run_dir.glob("*.safetensors"))

    summary = {
        "model_id": adapter.model_id,
        "resolution": adapter.resolution,
        "images": images,
        "depth_points": n_points,
        "bytes_per_image_per_point": round(per_point),
        "bytes_per_image_all_points": round(per_image),
        "shard_overhead": round(on_disk / written, 4),
        "images_per_second": round(images / elapsed, 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + json.dumps(summary, indent=2))
    print("\nExtrapolated cache, full patch tokens:")
    for label, count in (("ImageNet-100 (130k)", 130_000), ("ImageNet-1K (1.28M)", 1_280_000)):
        one = per_image * count
        print(f"  {label:22} {one / 1e12:7.2f} TB per model   {one * 6 / 1e12:7.2f} TB for the roster")

    tokens = adapter.model.config.image_size and (adapter.resolution // adapter.model.config.patch_size) ** 2
    print("\nSame at 130k images if semantic features are pooled to a grid:")
    for side in (8, 4):
        scaled = per_image * (side**2 / tokens) * 130_000
        print(f"  {side}x{side} grid{'':10} {scaled / 1e9:7.1f} GB per model   {scaled * 6 / 1e9:7.1f} GB for the roster")


if __name__ == "__main__":
    main()
