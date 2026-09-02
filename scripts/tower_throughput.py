#!/usr/bin/env python3
"""Measure one Tower adapter on a fixed image count with no competing workload."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vtb.extract import ADAPTERS, Collate
from vtb.images import ImageFolder


def configured_patch_size(adapter) -> int:
    value = adapter.model.config.patch_size
    if isinstance(value, (list, tuple)):
        if len(set(value)) != 1:
            raise ValueError(f"non-square patch size {value}")
        value = value[0]
    return int(value)


def consume_tower(adapter, inputs, image_ids) -> None:
    for batch in adapter.extract(inputs, image_ids):
        if batch.stage == "tower" and batch.layer_index == adapter.num_layers:
            return
    raise RuntimeError("adapter did not emit its final Tower Stage")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=768)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.count < 768:
        raise SystemExit("throughput comparisons require at least 768 measured images")

    adapter = ADAPTERS[args.model](resolution=args.resolution, device=args.device)
    patch_size = configured_patch_size(adapter)
    if args.resolution % patch_size:
        raise ValueError(f"resolution {args.resolution} is not divisible by patch size {patch_size}")
    tokens = (args.resolution // patch_size) ** 2

    dataset = ImageFolder(args.images, adapter.preprocess(), args.count)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=Collate(adapter.collate),
        prefetch_factor=4 if args.workers else None,
        persistent_workers=bool(args.workers),
    )

    warm = iter(loader)
    for _ in range(args.warmup_batches):
        inputs, image_ids = next(warm)
        consume_tower(adapter, inputs, image_ids)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    measured = 0
    started = time.perf_counter()
    for inputs, image_ids in loader:
        consume_tower(adapter, inputs, image_ids)
        measured += len(image_ids)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    result = {
        "model": args.model,
        "model_id": adapter.model_id,
        "resolution": args.resolution,
        "configured_patch_size": patch_size,
        "tower_tokens_per_image": tokens,
        "images": measured,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "warmup_batches": args.warmup_batches,
        "seconds": round(seconds, 3),
        "images_per_second": round(measured / seconds, 3),
        "scope": "preprocessing, transfer, Tower forward, and final Tower Stage materialization",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
