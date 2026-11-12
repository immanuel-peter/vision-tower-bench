from vtb.methods import descriptor as reducer_method
import argparse
import hashlib
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
    """Keep a class-stratified training subset. Validation and test stay full."""
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


@torch.inference_mode()
def predictions(model, features, index, device, batch_size=512):
    model.eval()
    return torch.cat([
        model(features[index[start:start + batch_size]].to(device)).argmax(dim=-1).cpu()
        for start in range(0, len(index), batch_size)
    ])


def run_cell(features, labels, split, num_classes, args) -> dict:
    best_lr, best_val = None, -1.0
    # Record each validation score to distinguish edge selections from plateaus.
    searched: dict[str, float] = {}
    unrounded = {}
    for lr in args.learning_rates:
        _, val, _ = train_once(
            features, labels, split, num_classes, args.readout, lr, 0,
            args.device, args.epochs, args.batch_size,
        )
        searched[f"{lr:g}"] = round(val, 4)
        unrounded[f"{lr:g}"] = val
        if val > best_val:
            best_lr, best_val = lr, val

    tests, saved_predictions = [], []
    for seed in range(args.seeds):
        model, _, test = train_once(
            features, labels, split, num_classes, args.readout, best_lr, seed,
            args.device, args.epochs, args.batch_size,
        )
        tests.append(test)
        if getattr(args, "save_predictions", False):
            saved_predictions.append(predictions(model, features, split.test, args.device).tolist())
        del model
    scores = torch.tensor(tests)
    result = {
        "learning_rate": best_lr,
        "learning_rate_search": searched,
        "val_accuracy": round(best_val, 4),
        "test_accuracy": round(scores.mean().item(), 4),
        "test_std": round(scores.std(unbiased=False).item(), 4),
        "seeds": args.seeds,
        "seed_ids": list(range(args.seeds)),
        "learning_rate_search_seed": 0,
        "test_accuracy_by_seed": tests,
    }
    if getattr(args, "save_predictions", False):
        result.update(test_predictions_by_seed=saved_predictions,
                      test_labels=labels[split.test].tolist(),
                      learning_rate_search_unrounded=unrounded,
                      validation_accuracy_unrounded=best_val,
                      fit_device=args.device, fit_threads=torch.get_num_threads())
    return result


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


def fit_reducer(
    features: torch.Tensor, split: Split, width: int, *, standardize_channels: bool = False
) -> probe.Reducer | probe.StandardizedReducer:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        reducer = probe.StandardizedReducer if standardize_channels else probe.Reducer
        return reducer.fit(features[split.train], width)


def feature_provenance(run: Path) -> dict:
    for name in ("provenance.json", "summary.json"):
        path = run / name
        if path.is_file():
            return {"cache_run": str(run), "metadata_file": name,
                    "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "details": json.loads(path.read_text())}
    return {"cache_run": str(run), "metadata_file": None}


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
    parser.add_argument("--standardize-channels", action="store_true", help="diagnostic: standardize training channels before matched PCA")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--save-predictions", action="store_true", help="retain selected-fit per-image class predictions for audited paired reuse")
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

    if args.standardize_channels and not args.match_capacity:
        raise ValueError("--standardize-channels requires --match-capacity")

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
            reducer = fit_reducer(features, split, args.width, standardize_channels=args.standardize_channels)
            features = reducer(features)

        result = run_cell(features, labels, split, num_classes, args)
        if args.save_predictions:
            result["test_image_ids"] = [image_ids[index] for index in split.test.tolist()]
        result |= {
            "method": reducer_method(standardize_channels=args.standardize_channels, matched=args.match_capacity),
            "model_id": meta["model_id"],
            "stage": stage,
            "layer_index": layer,
            "relative_depth": float(meta["relative_depth"]),
            "token_width": tokens.shape[-1],
            "trainable_parameters": probe.parameter_count(
                probe.build(args.readout, features.shape[-1], num_classes)
            ),
            "label_fraction": args.label_fraction,
            "standardize_channels": args.standardize_channels,
            "train_images": len(split.train),
            "full_train_images": len(full_split.train),
            "sample_ids_sha256": hashlib.sha256(json.dumps(image_ids).encode()).hexdigest(),
            "split_ids_sha256": {
                name: hashlib.sha256(json.dumps([image_ids[i] for i in getattr(split, name).tolist()]).encode()).hexdigest()
                for name in ("train", "val", "test")
            },
        }
        cells.append(result)
        print(
            f"{stage:>9} L{layer:02d}  depth {result['relative_depth']:.3f}  "
            f"top1 {result['test_accuracy']:.4f} +- {result['test_std']:.4f}  "
            f"lr {result['learning_rate']:g}  params {result['trainable_parameters']:,}",
            flush=True,
        )

    arm = "matched_standardized" if args.standardize_channels else "matched" if args.match_capacity else "raw"
    out = args.out or args.run / f"probe_{args.readout}_{arm}.json"
    out.write_text(json.dumps({
        "method": reducer_method(standardize_channels=args.standardize_channels, matched=args.match_capacity),
        "images": image_count,
        "feature_provenance": feature_provenance(args.run),
        "protocol": {"readout": args.readout, "capacity_matched": args.match_capacity,
                     "width": args.width, "classes": num_classes, "epochs": args.epochs,
                     "batch_size": args.batch_size, "device": args.device,
                     "learning_rates": args.learning_rates, "learning_rate_search_seed": 0,
                     "retraining_seeds": list(range(args.seeds))},
        "label_fraction": args.label_fraction,
        "standardize_channels": args.standardize_channels,
        "train_images": cells[0]["train_images"] if cells else 0,
        "full_train_images": cells[0]["full_train_images"] if cells else 0,
        "cells": cells,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
