import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from vtb.adapters.dinov2 import DINOv2Adapter
from vtb.adapters.moonvit_v2 import MoonViTV2Adapter
from vtb.images import ImageFolder

ADAPTERS = {"dinov2": DINOv2Adapter, "moonvit_v2": MoonViTV2Adapter}
IMAGENET_100 = 130_000


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frozen Tower features and measure cache growth.")
    ap.add_argument("--model", default="dinov2", choices=sorted(ADAPTERS))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("cache"))
    ap.add_argument("--limit", type=int, default=None, help="default is every image under --images")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--pool",
        type=int,
        default=None,
        metavar="SIDE",
        help="average the patch grid to SIDE x SIDE tokens; 4 for the semantic pillar, "
        "omit for the geometry pillar and the pooling validation subset (ADR-0005)",
    )
    args = ap.parse_args()

    adapter = ADAPTERS[args.model](resolution=args.resolution, device=args.device)
    dataset = ImageFolder(args.images, adapter.preprocess(), args.limit)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, collate_fn=Collate(adapter.collate)
    )

    tag = f"pool{args.pool}" if args.pool else "full"
    run_dir = args.out / f"{args.model}_{args.resolution}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(dataset)} images, {adapter.num_layers} layers, depth points {adapter.depth_points()}")
    print(f"grid: {tag}")

    written = images = tokens = slices = 0
    start = time.perf_counter()
    for shard, (inputs, image_ids) in enumerate(loader):
        slices = 0
        for batch in adapter.extract(inputs, image_ids):
            if args.pool:
                batch = batch.pooled(args.pool)
            batch.save(run_dir / f"{batch.stage}_L{batch.layer_index:02d}_{shard:05d}.safetensors")
            written += batch.nbytes
            slices += 1
            if batch.stage == "tower":
                tokens = batch.tokens.shape[1]
        images += len(image_ids)
        elapsed = time.perf_counter() - start
        print(f"{images}/{len(dataset)}  {images / elapsed:5.1f} img/s  {written / 1e9:6.2f} GB", flush=True)

    report(run_dir, adapter, args.pool, tokens, slices, images, written, time.perf_counter() - start)


@dataclass(frozen=True)
class Collate:
    """Splits the dataset samples from their image ids and lets the adapter batch them.

    A module-level class rather than a closure, because DataLoader workers pickle it.
    """

    batch: Callable

    def __call__(self, samples):
        values, image_ids = zip(*samples)
        return self.batch(list(values)), list(image_ids)


def report(run_dir, adapter, pool, tokens, slices, images, written, elapsed) -> None:
    per_image = written / images
    on_disk = sum(p.stat().st_size for p in run_dir.glob("*.safetensors"))

    summary = {
        "model_id": adapter.model_id,
        "resolution": adapter.resolution,
        "pooled_to": pool,
        "tokens_per_image": tokens,
        "images": images,
        "stages": list(adapter.stages),
        "depth_points": len(adapter.depth_points()),
        "slices_per_image": slices,
        "bytes_per_image_per_slice": round(per_image / slices),
        "bytes_per_image_all_points": round(per_image),
        "shard_overhead": round(on_disk / written, 4),
        "images_per_second": round(images / elapsed, 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2))

    total = per_image * IMAGENET_100
    print(f"\nImageNet-100 at {tokens} Tower tokens per image: {scale(total)} for this model.")
    print("  Roster totals need this run per model. Token width and Stage count both vary.")

    if pool is None:
        print("\nIf the semantic pillar pools the grid instead:")
        for side in (8, 4):
            small = per_image * (side**2 / tokens) * IMAGENET_100
            print(f"  {side}x{side}  {scale(small)} for this model")


def scale(nbytes: float) -> str:
    return f"{nbytes / 1e12:6.2f} TB" if nbytes >= 1e12 else f"{nbytes / 1e9:6.1f} GB"


if __name__ == "__main__":
    main()
