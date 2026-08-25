import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from vtb.cache import ShardWriter
from vtb.adapters.dinov2 import DINOv2Adapter
from vtb.adapters.moonvit_v2 import MoonViTV2Adapter
from vtb.images import ImageFolder

ADAPTERS = {"dinov2": DINOv2Adapter, "moonvit_v2": MoonViTV2Adapter}
IMAGENET_100 = 130_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen vision features and measure cache growth.")
    parser.add_argument("--model", default="dinov2", choices=sorted(ADAPTERS))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("cache"))
    parser.add_argument("--limit", type=int, default=None, help="default is every image under --images")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers. JPEG decode limits extraction, so set "
        "this near the vCPU count on a burst instance. At the default of 4 an A100 ran "
        "DINOv2 at 4 percent utilization.",
    )
    parser.add_argument(
        "--images-per-shard",
        type=int,
        default=512,
        help="images per output file. This keeps file count independent of --batch-size",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=None,
        metavar="SIDE",
        help="average the patch grid to SIDE x SIDE tokens; use 4 for semantic probes, "
        "omit for the geometry pillar and the pooling validation subset (ADR-0005)",
    )
    args = parser.parse_args()

    adapter = ADAPTERS[args.model](resolution=args.resolution, device=args.device)
    dataset = ImageFolder(args.images, adapter.preprocess(), args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=Collate(adapter.collate),
        prefetch_factor=4 if args.workers else None,
        persistent_workers=bool(args.workers),
    )

    tag = f"pool{args.pool}" if args.pool is not None else "full"
    run_dir = args.out / f"{args.model}_{args.resolution}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(dataset)} images, {adapter.num_layers} layers, depth points {adapter.depth_points()}")
    print(f"grid: {tag}")

    writer = ShardWriter(run_dir, args.images_per_shard)
    written_bytes = image_count = token_count = 0
    start = time.perf_counter()
    for inputs, image_ids in loader:
        for batch in adapter.extract(inputs, image_ids):
            if args.pool is not None:
                batch = batch.pooled(args.pool)
            writer.add(batch)
            written_bytes += batch.nbytes
            if batch.stage == "tower":
                token_count = batch.tokens.shape[1]
        image_count += len(image_ids)
        elapsed = time.perf_counter() - start
        print(
            f"{image_count}/{len(dataset)}  {image_count / elapsed:5.1f} img/s  "
            f"{written_bytes / 1e9:6.2f} GB",
            flush=True,
        )
    writer.close()

    report(
        run_dir,
        adapter,
        args.pool,
        token_count,
        writer.slice_count,
        image_count,
        written_bytes,
        time.perf_counter() - start,
    )


@dataclass(frozen=True)
class Collate:
    """Picklable adapter collator that separates image ids from model inputs."""

    batch: Callable

    def __call__(self, samples):
        values, image_ids = zip(*samples)
        return self.batch(list(values)), list(image_ids)


def report(run_dir, adapter, pool, tokens, slice_count, images, written_bytes, elapsed) -> None:
    per_image = written_bytes / images
    on_disk = sum(p.stat().st_size for p in run_dir.glob("*.safetensors"))

    summary = {
        "model_id": adapter.model_id,
        "resolution": adapter.resolution,
        "pooled_to": pool,
        "tokens_per_image": tokens,
        "images": images,
        "stages": list(adapter.stages),
        "depth_points": len(adapter.depth_points()),
        "slices_per_image": slice_count,
        "bytes_per_image_per_slice": round(per_image / slice_count),
        "bytes_per_image_all_points": round(per_image),
        "shard_overhead": round(on_disk / written_bytes, 4),
        "images_per_second": round(images / elapsed, 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2))

    total = per_image * IMAGENET_100
    print(f"\nImageNet-100 at {tokens} tower tokens per image: {scale(total)} for this model.")
    print("  Measure each model. Token width and stage count both vary.")

    if pool is None:
        print("\nIf the semantic pillar pools the grid instead:")
        for side in (8, 4):
            small = per_image * (side**2 / tokens) * IMAGENET_100
            print(f"  {side}x{side}  {scale(small)} for this model")


def scale(nbytes: float) -> str:
    return f"{nbytes / 1e12:6.2f} TB" if nbytes >= 1e12 else f"{nbytes / 1e9:6.1f} GB"


if __name__ == "__main__":
    main()
