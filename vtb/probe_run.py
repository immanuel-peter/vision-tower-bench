import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from vtb import cache, probe

# The eleven points PLAN.md specifies, extended below the old 3e-4 floor after the
# roster run found 107 of 112 attention cells selecting it.
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


@dataclass(frozen=True)
class Split:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


def split_indices(count: int, seed: int = 0, val: float = 0.15, test: float = 0.15) -> Split:
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    n_val, n_test = int(count * val), int(count * test)
    return Split(order[n_val + n_test:], order[:n_val], order[n_val:n_val + n_test])


def train_once(features, labels, split, num_classes, kind, lr, seed, device, epochs, batch_size):
    torch.manual_seed(seed)
    model = probe.build(kind, features.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for _ in range(epochs):
        model.train()
        order = split.train[torch.randperm(len(split.train))]
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            loss = nn.functional.cross_entropy(
                model(features[index].to(device)), labels[index].to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()

    return model, accuracy(model, features, labels, split.val, device), accuracy(
        model, features, labels, split.test, device
    )


@torch.inference_mode()
def accuracy(model, features, labels, index, device, batch_size=512) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        predicted = model(features[chunk].to(device)).argmax(dim=-1).cpu()
        correct += (predicted == labels[chunk]).sum().item()
    return correct / len(index)


def run_cell(features, labels, split, num_classes, args) -> dict:
    best_lr, best_val = None, -1.0
    # Record each validation score to distinguish edge selections from plateaus.
    searched: dict[str, float] = {}
    for lr in args.learning_rates:
        _, val, _ = train_once(
            features, labels, split, num_classes, args.readout, lr, 0,
            args.device, args.epochs, args.batch_size,
        )
        searched[f"{lr:g}"] = round(val, 4)
        if val > best_val:
            best_lr, best_val = lr, val

    tests = []
    for seed in range(args.seeds):
        _, _, test = train_once(
            features, labels, split, num_classes, args.readout, best_lr, seed,
            args.device, args.epochs, args.batch_size,
        )
        tests.append(test)
    scores = torch.tensor(tests)
    return {
        "learning_rate": best_lr,
        "learning_rate_search": searched,
        "val_accuracy": round(best_val, 4),
        "test_accuracy": round(scores.mean().item(), 4),
        "test_std": round(scores.std(unbiased=False).item(), 4),
        "seeds": args.seeds,
    }


def selected_slices(run: Path, depth_points: list[int] | None) -> list[tuple[str, int]]:
    slices = cache.slices(run)
    if depth_points is None:
        return slices
    selected = set(depth_points)
    return [(stage, layer) for stage, layer in slices if layer in selected]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a semantic readout on cached features.")
    parser.add_argument("--run", type=Path, required=True, help="cache directory written by vtb.extract")
    parser.add_argument("--labels", type=Path, required=True, help="JSON written by scripts/export_imagenet100.py")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--readout", default="attention", choices=("attention", "mean"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--match-capacity",
        action="store_true",
        help="fit a frozen PCA reduction to --width first, so every cell trains the same "
        "number of parameters regardless of token width",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(LEARNING_RATES),
    )
    parser.add_argument(
        "--depth-points",
        type=int,
        nargs="+",
        default=None,
        help="only probe cache slices with these Tower layer indices (default: all)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    label_map = json.loads(args.labels.read_text())["labels"]
    num_classes = len(set(label_map.values()))

    cells = []
    image_count = 0
    for stage, layer in selected_slices(args.run, args.depth_points):
        tokens, image_ids, meta = cache.load(args.run, stage, layer)
        image_count = len(image_ids)
        labels = torch.tensor([label_map[i] for i in image_ids])
        split = split_indices(len(image_ids))

        features = tokens.float()
        if args.match_capacity:
            reducer = probe.Reducer.fit(features[split.train], args.width)
            features = reducer(features)

        result = run_cell(features, labels, split, num_classes, args)
        result |= {
            "model_id": meta["model_id"],
            "stage": stage,
            "layer_index": layer,
            "relative_depth": float(meta["relative_depth"]),
            "token_width": tokens.shape[-1],
            "trainable_parameters": probe.parameter_count(
                probe.build(args.readout, features.shape[-1], num_classes)
            ),
        }
        cells.append(result)
        print(
            f"{stage:>9} L{layer:02d}  depth {result['relative_depth']:.3f}  "
            f"top1 {result['test_accuracy']:.4f} +- {result['test_std']:.4f}  "
            f"lr {result['learning_rate']:g}  params {result['trainable_parameters']:,}",
            flush=True,
        )

    out = args.out or args.run / f"probe_{args.readout}_{'matched' if args.match_capacity else 'raw'}.json"
    out.write_text(json.dumps({"images": image_count, "cells": cells}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
