"""2-layer GELU MLP on mean-pooled MiniMax-M3 full-grid tokens.

Same split / LR grid / 20 epochs / 3 seeds as `vtb.probe_run`. Mean-pools
per shard so the full-grid cache never sits in RAM as float32.
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

from vtb import cache, probe
from vtb.probe_run import LEARNING_RATES, run_cell, split_indices


class MeanMLP(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1) if tokens.ndim == 3 else tokens
        return self.net(pooled)


def load_mean_pooled(run_dir: Path, stage: str, layer: int) -> tuple[torch.Tensor, list[str], dict]:
    parts: list[torch.Tensor] = []
    image_ids: list[str] = []
    metadata: dict = {}
    for path in cache.shards(run_dir, stage, layer):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            tokens = handle.get_tensor("tokens")
            parts.append(tokens.float().mean(dim=1, keepdim=True))
            image_ids.extend(metadata["image_ids"].split("\n"))
    return torch.cat(parts), image_ids, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("cache/minimax_m3_448_full"))
    parser.add_argument("--labels", type=Path, default=Path("data/imagenet100/validation_labels.json"))
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--stage", default="tower")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("/tmp/mm3_mlp.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    tokens, image_ids, meta = load_mean_pooled(args.run, args.stage, args.layer)
    label_map = json.loads(args.labels.read_text())["labels"]
    labels = torch.tensor([label_map[i] for i in image_ids])
    split = split_indices(len(image_ids))
    features = tokens.float()
    num_classes = len(set(label_map.values()))
    dim = features.shape[-1]

    original = probe.build

    def build(kind: str, d: int, n: int, width: int = 512) -> nn.Module:
        if kind == "mlp":
            return MeanMLP(d, n, hidden=args.hidden)
        return original(kind, d, n, width)

    probe.build = build
    try:
        result = run_cell(
            features,
            labels,
            split,
            num_classes,
            Namespace(
                readout="mlp",
                learning_rates=list(LEARNING_RATES),
                device=args.device,
                epochs=20,
                batch_size=256,
                seeds=3,
            ),
        )
    finally:
        probe.build = original

    result |= {
        "model_id": meta.get("model_id"),
        "stage": args.stage,
        "layer_index": args.layer,
        "readout": "mlp_mean",
        "hidden": args.hidden,
        "token_width": dim,
        "n_tokens": 1,
        "train_images": len(split.train),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"mlp L{args.layer} top1 {result['test_accuracy']:.4f} +- {result['test_std']:.4f}  "
        f"lr {result['learning_rate']:g}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
