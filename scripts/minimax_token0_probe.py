"""Probe MiniMax-M3 full-grid L32 with token 0 alone vs token 0 removed.

Loads the full-grid cache written by `python -m vtb.extract` (no --pool),
keeps only `tokens[:, :1, :]` or `tokens[:, 1:, :]`, and trains the standard
attention and mean cells via `vtb.probe_run.run_cell` (same LR grid, 20
epochs, 3 seeds, match-capacity PCA to 512).
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import torch
from safetensors import safe_open

from vtb import cache, probe
from vtb.probe_run import LEARNING_RATES, fit_reducer, run_cell, split_indices


def load_token_view(
    run_dir: Path, stage: str, layer: int, kind: str
) -> tuple[torch.Tensor, list[str], dict[str, str]]:
    """Load one cache slice, keeping token 0, dropping it, or keeping all."""
    parts: list[torch.Tensor] = []
    image_ids: list[str] = []
    metadata: dict[str, str] = {}
    for path in cache.shards(run_dir, stage, layer):
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            tokens = handle.get_tensor("tokens")
            if kind == "token0":
                tokens = tokens[:, :1, :].contiguous()
            elif kind == "no_token0":
                tokens = tokens[:, 1:, :].contiguous()
            elif kind != "all":
                raise ValueError(kind)
            parts.append(tokens)
            image_ids.extend(metadata["image_ids"].split("\n"))
    if not parts:
        raise FileNotFoundError(f"no shards for {stage} L{layer:02d} under {run_dir}")
    return torch.cat(parts), image_ids, metadata


def cell_args(readout: str, device: str) -> Namespace:
    return Namespace(
        readout=readout,
        learning_rates=list(LEARNING_RATES),
        device=device,
        epochs=20,
        batch_size=256,
        seeds=3,
    )


def probe_view(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    split,
    num_classes: int,
    readout: str,
    device: str,
    match_capacity: bool,
    width: int,
) -> dict:
    features = tokens.float()
    if match_capacity:
        reducer = fit_reducer(features, split, width)
        features = reducer(features)
    result = run_cell(features, labels, split, num_classes, cell_args(readout, device))
    result["token_width"] = int(tokens.shape[-1])
    result["n_tokens"] = int(tokens.shape[1])
    result["trainable_parameters"] = probe.parameter_count(
        probe.build(readout, features.shape[-1], num_classes)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("cache/minimax_m3_448_full"))
    parser.add_argument("--labels", type=Path, default=Path("data/imagenet100/validation_labels.json"))
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--stage", default="tower")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("/tmp/mm3_token0.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    label_map = json.loads(args.labels.read_text())["labels"]
    num_classes = len(set(label_map.values()))
    cells = []

    for kind in ("token0", "no_token0"):
        print(f"\n=== loading {kind} ===", flush=True)
        tokens, image_ids, meta = load_token_view(args.run, args.stage, args.layer, kind)
        labels = torch.tensor([label_map[i] for i in image_ids])
        split = split_indices(len(image_ids))
        print(
            f"{kind}: tokens {tuple(tokens.shape)} dtype={tokens.dtype} "
            f"nbytes={tokens.nbytes / 1e9:.2f} GB",
            flush=True,
        )
        for readout in ("attention", "mean"):
            print(f"--- {kind} {readout} matched ---", flush=True)
            result = probe_view(
                tokens, labels, split, num_classes, readout, args.device, True, args.width
            )
            result |= {
                "model_id": meta.get("model_id"),
                "stage": args.stage,
                "layer_index": args.layer,
                "relative_depth": float(meta.get("relative_depth", "1.0")),
                "view": kind,
                "readout": readout,
                "match_capacity": True,
                "train_images": len(split.train),
            }
            cells.append(result)
            print(
                f"{kind:>10} {readout:>9}  top1 {result['test_accuracy']:.4f} "
                f"+- {result['test_std']:.4f}  lr {result['learning_rate']:g}  "
                f"tokens {result['n_tokens']}",
                flush=True,
            )
        del tokens

    payload = {
        "images": len(image_ids),
        "run": str(args.run),
        "layer": args.layer,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "cells": cells,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")


if __name__ == "__main__":
    main()
