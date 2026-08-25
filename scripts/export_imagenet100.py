import argparse
import json
from pathlib import Path

from datasets import load_dataset

REPO = "ilee0022/ImageNet100"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", type=Path, default=Path("data/imagenet100"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(REPO, split=args.split)
    if args.limit:
        dataset = dataset.select(range(args.limit))

    labels, names = {}, {}
    for i, example in enumerate(dataset):
        image_id = f"{args.split}_{i:06d}"
        example["image"].convert("RGB").save(out / f"{image_id}.jpg", quality=95)
        labels[image_id] = example["label"]
        names.setdefault(example["label"], example["text"])

    payload = {"repo": REPO, "split": args.split, "labels": labels, "class_names": names}
    (args.out / f"{args.split}_labels.json").write_text(json.dumps(payload) + "\n")
    print(f"{len(labels)} images, {len(names)} classes -> {out}")


if __name__ == "__main__":
    main()
