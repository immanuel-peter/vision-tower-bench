#!/usr/bin/env python3
"""Extract final-Stage caches for every Perturbation Study condition."""

import argparse
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader

from vtb.cache import ShardWriter
from vtb.extract import ADAPTERS, Collate
from vtb.images import ImageFolder


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--identity-images", type=Path, required=True)
    ap.add_argument("--condition-images", type=Path, required=True)
    ap.add_argument("--transform-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=2_000)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--images-per-shard", type=int, default=512)
    return ap


def main() -> None:
    args = parser().parse_args()
    protocol = json.loads(args.transform_manifest.read_text())
    if args.count != protocol["images"]:
        raise SystemExit(f"requested {args.count} images but manifest has {protocol['images']}")

    adapter = ADAPTERS[args.model](resolution=args.resolution, device=args.device)
    summaries = []
    for condition in protocol["conditions"]:
        image_root = args.identity_images if condition == "identity" else args.condition_images / condition
        dataset = ImageFolder(image_root, adapter.preprocess(), args.count)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=Collate(adapter.collate),
            prefetch_factor=4 if args.workers else None,
            persistent_workers=bool(args.workers),
        )
        run_dir = args.out / condition / f"{args.model}_{args.resolution}_pool4"
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = ShardWriter(run_dir, args.images_per_shard)
        image_count = written = 0
        started = time.perf_counter()
        for inputs, image_ids in loader:
            for batch in adapter.extract(inputs, image_ids):
                if batch.layer_index != adapter.num_layers:
                    continue
                batch = batch.pooled(4)
                writer.add(batch)
                written += batch.nbytes
            image_count += len(image_ids)
        writer.close()
        seconds = time.perf_counter() - started
        summary = {
            "condition": condition,
            "model": args.model,
            "model_id": adapter.model_id,
            "images": image_count,
            "stages": list(adapter.stages),
            "layer_index": adapter.num_layers,
            "pooled_to": 4,
            "bytes": written,
            "seconds": round(seconds, 3),
            "images_per_second": round(image_count / seconds, 3),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(f"{condition}: {image_count} images at {image_count / seconds:.2f} img/s", flush=True)

    report = args.out / f"{args.model}_extraction.json"
    report.write_text(json.dumps({"model": args.model, "conditions": summaries}, indent=2) + "\n")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
