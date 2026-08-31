#!/usr/bin/env python3
"""Train two selected semantic cells and write a paired image bootstrap dataset."""

import argparse
import json
from pathlib import Path

import torch

from vtb import cache, probe_run
from vtb.semantic_bootstrap import paired_image_bootstrap


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-run", type=Path, required=True)
    ap.add_argument("--first-name", required=True)
    ap.add_argument("--first-layer", type=int, required=True)
    ap.add_argument("--second-run", type=Path, required=True)
    ap.add_argument("--second-name", required=True)
    ap.add_argument("--second-layer", type=int, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--readout", choices=("attention", "mean"), default="attention")
    ap.add_argument("--arm", choices=("raw", "matched"), default="matched")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--learning-rates", type=float, nargs="+", default=list(probe_run.LEARNING_RATES)
    )
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    return ap


@torch.inference_mode()
def predict(model, features, index, device, batch_size=512) -> torch.Tensor:
    model.eval()
    predictions = []
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        predictions.append(model(features[chunk].to(device)).argmax(dim=-1).cpu())
    return torch.cat(predictions)


def train_cell(run, layer, labels_by_id, args):
    tokens, image_ids, metadata = cache.load(run, "tower", layer)
    labels = torch.tensor([labels_by_id[image_id] for image_id in image_ids])
    split = probe_run.split_indices(len(image_ids))
    features = tokens.float()
    if args.arm == "matched":
        reducer = probe_run.fit_reducer(features, split, args.width)
        features = reducer(features)
    test_labels = labels[split.test]
    selected_rate = None
    best_validation = -1.0
    search = {}
    for rate in args.learning_rates:
        _, validation, _ = probe_run.train_once(
            features,
            labels,
            split,
            len(set(labels_by_id.values())),
            args.readout,
            rate,
            0,
            args.device,
            args.epochs,
            args.batch_size,
        )
        search[f"{rate:g}"] = validation
        if validation > best_validation:
            selected_rate = rate
            best_validation = validation

    predictions = []
    accuracies = []
    for seed in range(args.seeds):
        model, _, _ = probe_run.train_once(
            features,
            labels,
            split,
            len(set(labels_by_id.values())),
            args.readout,
            selected_rate,
            seed,
            args.device,
            args.epochs,
            args.batch_size,
        )
        predicted = predict(model, features, split.test, args.device)
        predictions.append(predicted)
        accuracies.append((predicted == test_labels).float().mean().item())
    return {
        "image_ids": [image_ids[index] for index in split.test.tolist()],
        "labels": test_labels,
        "predictions": torch.stack(predictions),
        "accuracies": accuracies,
        "model_id": metadata["model_id"],
        "relative_depth": float(metadata["relative_depth"]),
        "learning_rate": selected_rate,
        "learning_rate_search": search,
        "validation_accuracy": best_validation,
    }


def main() -> None:
    args = parser().parse_args()
    labels_by_id = json.loads(args.labels.read_text())["labels"]
    first = train_cell(args.first_run, args.first_layer, labels_by_id, args)
    second = train_cell(args.second_run, args.second_layer, labels_by_id, args)
    if first["image_ids"] != second["image_ids"]:
        raise ValueError("the two cells do not have the same paired test images")
    if not torch.equal(first["labels"], second["labels"]):
        raise ValueError("the two cells do not have the same paired test labels")

    first_correct = first["predictions"] == first["labels"]
    second_correct = second["predictions"] == second["labels"]
    interval = paired_image_bootstrap(
        first_correct,
        second_correct,
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.bootstrap_seed,
    )
    rows = []
    for position, image_id in enumerate(first["image_ids"]):
        rows.append(
            {
                "image_id": image_id,
                "label": first["labels"][position].item(),
                args.first_name: first["predictions"][:, position].tolist(),
                args.second_name: second["predictions"][:, position].tolist(),
            }
        )
    payload = {
        "protocol": {
            "readout": args.readout,
            "arm": args.arm,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "test_images": len(rows),
            "pairing": "seed-mean correctness, paired over test images",
        },
        "first": {
            "name": args.first_name,
            "model_id": first["model_id"],
            "layer_index": args.first_layer,
            "relative_depth": first["relative_depth"],
            "learning_rate": first["learning_rate"],
            "learning_rate_search": first["learning_rate_search"],
            "validation_accuracy": first["validation_accuracy"],
            "seed_accuracies": first["accuracies"],
            "mean_accuracy": sum(first["accuracies"]) / len(first["accuracies"]),
        },
        "second": {
            "name": args.second_name,
            "model_id": second["model_id"],
            "layer_index": args.second_layer,
            "relative_depth": second["relative_depth"],
            "learning_rate": second["learning_rate"],
            "learning_rate_search": second["learning_rate_search"],
            "validation_accuracy": second["validation_accuracy"],
            "seed_accuracies": second["accuracies"],
            "mean_accuracy": sum(second["accuracies"]) / len(second["accuracies"]),
        },
        "difference": interval,
        "predictions": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    summary = {key: value for key, value in payload.items() if key != "predictions"}
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
