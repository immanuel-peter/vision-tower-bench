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


def subsample_train(
    split: Split,
    labels: torch.Tensor,
    fraction: float,
    seed: int = 0,
) -> Split:
    """Apply an exact, class-stratified budget to the training indices only.

    The requested count is rounded from the full training split. When that count is
    smaller than the number of represented classes, one example per class is retained.
    Validation and test indices are returned unchanged.
    """
    if not 0 < fraction <= 1:
        raise ValueError("label fraction must be in (0, 1]")
    if fraction == 1:
        return split

    train_labels = labels[split.train]
    classes, sizes = torch.unique(train_labels, sorted=True, return_counts=True)
    target = max(len(classes), round(len(split.train) * fraction))
    target = min(len(split.train), target)

    exact = sizes.float() * (target / len(split.train))
    quota = exact.floor().long().clamp_min(1)
    quota = torch.minimum(quota, sizes)
    while int(quota.sum()) < target:
        available = quota < sizes
        priority = exact - quota.float()
        priority[~available] = -torch.inf
        quota[priority.argmax()] += 1

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for class_id, count in zip(classes, quota):
        candidates = split.train[train_labels == class_id]
        order = torch.randperm(len(candidates), generator=generator)
        selected.append(candidates[order[:int(count)]])
    train = torch.cat(selected)
    train = train[torch.randperm(len(train), generator=generator)]
    return Split(train, split.val, split.test)


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


def selected_slices(
    run: Path,
    depth_points: list[int] | None,
    stages: list[str] | None = None,
) -> list[tuple[str, int]]:
    slices = cache.slices(run)
    selected_depths = set(depth_points) if depth_points is not None else None
    selected_stages = set(stages) if stages is not None else None
    return [
        (stage, layer) for stage, layer in slices
        if (selected_depths is None or layer in selected_depths)
        and (selected_stages is None or stage in selected_stages)
    ]


def fit_reducer(features: torch.Tensor, split: Split, width: int) -> probe.Reducer:
    # torch.svd_lowrank is randomized. Seed it per cell without changing the RNG state
    # used by the readout, so a --depth-points invocation matches a matrix invocation.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return probe.Reducer.fit(features[split.train], width)


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
        "--label-fraction",
        type=float,
        default=1.0,
        help="class-stratified fraction of the training split to label; validation and "
        "test splits remain unchanged",
    )
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
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("tower", "merged", "projected"),
        default=None,
        help="only probe these Stages (default: all available Stages)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    label_map = json.loads(args.labels.read_text())["labels"]
    num_classes = len(set(label_map.values()))

    cells = []
    image_count = 0
    for stage, layer in selected_slices(args.run, args.depth_points, args.stages):
        tokens, image_ids, meta = cache.load(args.run, stage, layer)
        image_count = len(image_ids)
        labels = torch.tensor([label_map[i] for i in image_ids])
        full_split = split_indices(len(image_ids))
        split = subsample_train(full_split, labels, args.label_fraction)

        features = tokens.float()
        if args.match_capacity:
            reducer = fit_reducer(features, split, args.width)
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
            "label_fraction": args.label_fraction,
            "train_images": len(split.train),
            "full_train_images": len(full_split.train),
        }
        cells.append(result)
        print(
            f"{stage:>9} L{layer:02d}  depth {result['relative_depth']:.3f}  "
            f"top1 {result['test_accuracy']:.4f} +- {result['test_std']:.4f}  "
            f"lr {result['learning_rate']:g}  params {result['trainable_parameters']:,}",
            flush=True,
        )

    out = args.out or args.run / f"probe_{args.readout}_{'matched' if args.match_capacity else 'raw'}.json"
    out.write_text(json.dumps({
        "images": image_count,
        "label_fraction": args.label_fraction,
        "train_images": cells[0]["train_images"] if cells else 0,
        "full_train_images": cells[0]["full_train_images"] if cells else 0,
        "cells": cells,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
