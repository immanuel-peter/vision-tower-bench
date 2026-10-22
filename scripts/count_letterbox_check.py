#!/usr/bin/env python3
"""Run saved FSC-147 count readouts, frozen, on full-frame letterboxed FSC-147 validation images.

This tests the CountQA transfer preprocessing on data with known counts before
any CountQA result exists. The letterbox count is every release dot, the crop
count only the dots inside the centre square, so compare errors, not counts.
Nothing here is fit or selected; the heads, PCA and preprocessing are frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vtb import cache, counting, geometry_run
from vtb.count_metrics import bins_for, evaluate_count, summarise


def crop_val(cells_dir: Path, key: str, seed: int) -> dict | None:
    """The same readout's centre-crop validation errors, from the runner's saved rows."""
    path = cells_dir / f"{key}.val.tsv"
    if not path.exists():
        return None
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:]]
    errors = np.array([[float(r[9]), float(r[10]), float(r[8])] for r in rows if int(r[5]) == seed])
    return {"mae": round(errors[:, 0].mean(), 4), "rmse": round(float(np.sqrt(errors[:, 1].mean())), 4),
            "bias": round(errors[:, 2].mean(), 4), "images": len(errors)}


@torch.inference_mode()
def evaluate(readout, tokens, ids, images, boxes_per_image, device, batch_size) -> dict:
    counts, rects = [], []
    totals: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        rects = [tuple(images[i]["valid"]) for i in chunk]
        boxes = None
        if readout.conditioned:
            boxes = [np.asarray([b for b, ok in zip(images[i]["boxes"], images[i]["box_valid"]) if ok][:boxes_per_image])
                     for i in chunk]
        density = readout.density(tokens[start:start + len(chunk)].to(device), boxes, rects)
        valid = torch.from_numpy(np.stack([counting.coverage(r, counting.TARGET_GRID) for r in rects]))
        metrics = evaluate_count(density.cpu(), torch.tensor([images[i]["count"] for i in chunk]), valid)
        for key, value in metrics.items():
            totals.setdefault(key, []).append(value)
    metrics = {key: torch.cat(values) for key, values in totals.items()}
    return summarise(metrics, bins_for(metrics["true_count"].tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True, help="letterbox-val cache for one model")
    ap.add_argument("--manifest", type=Path, required=True, help="letterbox_val_manifest.json")
    ap.add_argument("--readouts", type=Path, required=True, help="readouts/<protocol>/<model>")
    ap.add_argument("--results", type=Path, required=True,
                    help="directory holding <model>_count_<arm>_<capacity>.cells from the FSC run")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    images = json.loads(args.manifest.read_text())["images"]
    completion = cache.completion(args.features)
    out = {"protocol": counting.PROTOCOL, "name": args.name, "frame": "letterbox",
           "fill": list(counting.LETTERBOX_FILL), "features": str(args.features),
           "features_complete_sha256": completion["sha256"],
           "note": "transfer diagnostic on FSC-147 val; frozen readouts; never used for selection",
           "readouts": []}
    for directory in sorted(p for p in args.readouts.glob("*_*/*_L*") if (p / "seed0.safetensors").exists()):
        arm_capacity, key = directory.parent.name, directory.name
        stage, layer = key.rsplit("_L", 1)
        batch = cache.load_batch(args.features, stage, int(layer))
        ids = [i for i in batch.image_ids if i in images and images[i]["exemplar_eligible"]]
        rows = [batch.image_ids.index(i) for i in ids]
        tokens = batch.tokens[rows]
        entry = {"readout": f"{arm_capacity}/{key}", "images": len(ids), "seeds": []}
        for seed in args.seeds:
            readout = geometry_run.load_readout(directory, seed, args.device)
            geometry_run.check_readout(readout, batch)
            row = {"seed": seed, "centre_crop_val": crop_val(args.results / f"{args.name}_count_{arm_capacity}.cells", key, seed)}
            if readout.conditioned:
                row["letterbox_three_boxes"] = evaluate(readout, tokens, ids, images, 3, args.device, args.batch_size)
                row["letterbox_one_box"] = evaluate(readout, tokens, ids, images, 1, args.device, args.batch_size)
            else:
                row["letterbox"] = evaluate(readout, tokens, ids, images, 0, args.device, args.batch_size)
            entry["seeds"].append(row)
            print(args.name, entry["readout"], seed, json.dumps({k: v.get("mae") for k, v in row.items()
                                                               if isinstance(v, dict)}), flush=True)
        out["readouts"].append(entry)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
