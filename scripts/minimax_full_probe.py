"""Memory-efficient full-grid probe for MiniMax-M3.

`vtb.probe_run` materializes the whole slice as float32 before PCA
(13k x 1024 x 1280 x 4 ≈ 68 GB), which OOMs on a 70 GB box. This script
keeps the official split / LR grid / 20 epochs / 3 seeds / match-capacity
512 PCA, but:

- fits the reducer on a seeded subsample of train tokens
- projects shards in chunks
- calls `vtb.probe_run.run_cell` on the reduced features

Use only when vanilla `python -m vtb.probe_run` OOMs on a full-grid cache.
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
from vtb.probe_run import LEARNING_RATES, run_cell, split_indices


def apply_view(tokens: torch.Tensor, view: str, mean_pool: bool) -> torch.Tensor:
    if view == "token0":
        tokens = tokens[:, :1, :].contiguous()
    elif view == "no_token0":
        tokens = tokens[:, 1:, :].contiguous()
    elif view != "all":
        raise ValueError(view)
    if mean_pool:
        tokens = tokens.float().mean(dim=1, keepdim=True)
    return tokens


def iter_shards(run_dir: Path, stage: str, layer: int, view: str = "all", mean_pool: bool = False):
    for path in cache.shards(run_dir, stage, layer):
        with safe_open(path, framework="pt") as handle:
            meta = handle.metadata() or {}
            tokens = apply_view(handle.get_tensor("tokens"), view, mean_pool)
            ids = meta["image_ids"].split("\n")
            yield tokens, ids, meta


def count_and_ids(
    run_dir: Path, stage: str, layer: int, view: str = "all", mean_pool: bool = False
) -> tuple[int, list[str], dict]:
    ids: list[str] = []
    meta: dict = {}
    for _, image_ids, metadata in iter_shards(run_dir, stage, layer, view, mean_pool):
        ids.extend(image_ids)
        meta = metadata
    return len(ids), ids, meta


def fit_reducer_subsample(
    run_dir: Path,
    stage: str,
    layer: int,
    train_index: torch.Tensor,
    width: int,
    max_tokens: int = 1_000_000,
    view: str = "all",
    mean_pool: bool = False,
) -> probe.Reducer:
    """Fit PCA on a seeded subset of train-split tokens, streaming shards."""
    train_set = set(int(i) for i in train_index.tolist())
    collected: list[torch.Tensor] = []
    n_tokens = 0
    offset = 0
    g = torch.Generator().manual_seed(0)
    for tokens, ids, _ in iter_shards(run_dir, stage, layer, view, mean_pool):
        n = tokens.shape[0]
        keep = [i for i, _ in enumerate(range(offset, offset + n)) if (offset + i) in train_set]
        offset += n
        if not keep:
            continue
        sl = tokens[keep].float().reshape(-1, tokens.shape[-1])
        # Subsample rows inside the shard so we never hold 9e6 tokens.
        if sl.shape[0] > 8192:
            perm = torch.randperm(sl.shape[0], generator=g)[:8192]
            sl = sl[perm]
        collected.append(sl)
        n_tokens += sl.shape[0]
        if n_tokens >= max_tokens:
            break
    flat = torch.cat(collected)
    if flat.shape[0] > max_tokens:
        perm = torch.randperm(flat.shape[0], generator=g)[:max_tokens]
        flat = flat[perm]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return probe.Reducer.fit(flat.unsqueeze(0), width)


def project_all(
    run_dir: Path,
    stage: str,
    layer: int,
    reducer: probe.Reducer | None,
    view: str = "all",
    mean_pool: bool = False,
) -> torch.Tensor:
    parts = []
    for tokens, _, _ in iter_shards(run_dir, stage, layer, view, mean_pool):
        feats = tokens.float()
        if reducer is not None:
            feats = reducer(feats)
        parts.append(feats)
    return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--readout", default="attention", choices=("attention", "mean"))
    parser.add_argument("--match-capacity", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", nargs="+", default=["tower"])
    parser.add_argument("--depth-points", type=int, nargs="+", default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--token-view", default="all", choices=("all", "token0", "no_token0"))
    parser.add_argument(
        "--mean-pool-first",
        action="store_true",
        help="average tokens per image before the readout (cheap full-grid mean)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    label_map = json.loads(args.labels.read_text())["labels"]
    num_classes = len(set(label_map.values()))
    cells = []
    image_count = 0

    slices = cache.slices(args.run)
    selected_depths = set(args.depth_points) if args.depth_points is not None else None
    selected_stages = set(args.stages) if args.stages is not None else None
    chosen = [
        (stage, layer)
        for stage, layer in slices
        if (selected_depths is None or layer in selected_depths)
        and (selected_stages is None or stage in selected_stages)
    ]

    cell_ns = Namespace(
        readout=args.readout,
        learning_rates=list(LEARNING_RATES),
        device=args.device,
        epochs=20,
        batch_size=256,
        seeds=3,
    )

    for stage, layer in chosen:
        print(f"loading {stage} L{layer:02d} view={args.token_view} mean_pool={args.mean_pool_first}", flush=True)
        n, image_ids, meta = count_and_ids(
            args.run, stage, layer, args.token_view, args.mean_pool_first
        )
        image_count = n
        labels = torch.tensor([label_map[i] for i in image_ids])
        split = split_indices(n)
        reducer = None
        if args.match_capacity:
            print(f"  fitting subsampled PCA to {args.width}", flush=True)
            reducer = fit_reducer_subsample(
                args.run, stage, layer, split.train, args.width,
                view=args.token_view, mean_pool=args.mean_pool_first,
            )
        features = project_all(
            args.run, stage, layer, reducer, args.token_view, args.mean_pool_first
        )
        print(f"  features {tuple(features.shape)} {features.nbytes / 1e9:.2f} GB", flush=True)
        result = run_cell(features, labels, split, num_classes, cell_ns)
        result |= {
            "model_id": meta.get("model_id"),
            "stage": stage,
            "layer_index": layer,
            "relative_depth": float(meta.get("relative_depth", "nan")),
            "token_width": int(features.shape[-1]) if args.match_capacity else int(meta.get("token_width", features.shape[-1])),
            "trainable_parameters": probe.parameter_count(
                probe.build(args.readout, features.shape[-1], num_classes)
            ),
            "train_images": len(split.train),
            "pca": "subsampled_train_tokens" if args.match_capacity else "none",
            "token_view": args.token_view,
            "mean_pool_first": args.mean_pool_first,
        }
        cells.append(result)
        print(
            f"{stage:>9} L{layer:02d}  top1 {result['test_accuracy']:.4f} "
            f"+- {result['test_std']:.4f}  lr {result['learning_rate']:g}",
            flush=True,
        )
        del features, reducer

    payload = {
        "images": image_count,
        "cells": cells,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "note": "PCA fitted on a seeded subsample of train tokens to fit in 70 GB RAM",
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
