#!/usr/bin/env python3
"""Run one Tower on a Probe3D correspondence dataset without training a probe."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nn_F
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_F

from vtb import correspondence
from vtb.extract import ADAPTERS


@dataclass(frozen=True)
class Frame:
    scale_x: float
    scale_y: float
    left: int
    top: int
    resolution: int

    def points(self, xy: torch.Tensor) -> torch.Tensor:
        out = xy.float().clone()
        out[:, 0] = out[:, 0] * self.scale_x - self.left
        out[:, 1] = out[:, 1] * self.scale_y - self.top
        return out

    def intrinsics(self, matrix: torch.Tensor) -> torch.Tensor:
        out = matrix.float().clone()
        out[0] *= self.scale_x
        out[1] *= self.scale_y
        out[0, 2] -= self.left
        out[1, 2] -= self.top
        return out

    def target(self, values: torch.Tensor) -> torch.Tensor:
        height = round(values.shape[-2] * self.scale_y)
        width = round(values.shape[-1] * self.scale_x)
        resized = nn_F.interpolate(
            values[None, None].float(), (height, width), mode="nearest"
        )[0, 0]
        return resized[self.top:self.top + self.resolution, self.left:self.left + self.resolution]


def canonical(image: Image.Image, resolution: int) -> tuple[Image.Image, Frame]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    resized = tv_F.resize(image, resolution, InterpolationMode.BICUBIC, antialias=True)
    width, height = resized.size
    left = int(round((width - resolution) / 2.0))
    top = int(round((height - resolution) / 2.0))
    cropped = tv_F.center_crop(resized, [resolution, resolution])
    return cropped, Frame(
        width / original_width,
        height / original_height,
        left,
        top,
        resolution,
    )


def read_depth(path: Path, divisor: float = 1000.0) -> torch.Tensor:
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image).astype(np.float32)) / divisor


def align_rgb_to_depth(image: Image.Image, depth: torch.Tensor) -> Image.Image:
    size = [depth.shape[-2], depth.shape[-1]]
    if image.size == (size[1], size[0]):
        return image
    return tv_F.resize(image, size, InterpolationMode.BICUBIC, antialias=True)


def navi_depth(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        disparity = np.array(image).astype(np.uint16).astype(np.float32)
    disparity /= (2**16 - 1) * 10.0
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = 1.0 / disparity[valid] / 1000.0
    return torch.from_numpy(depth)


def quaternion_matrix(q) -> torch.Tensor:
    q = torch.as_tensor(q, dtype=torch.float32)
    w, x, y, z = q
    scale = 2.0 / q.square().sum()
    rotation = torch.tensor([
        [1 - scale * (y*y + z*z), scale * (x*y - z*w), scale * (x*z + y*w)],
        [scale * (x*y + z*w), 1 - scale * (x*x + z*z), scale * (y*z - x*w)],
        [scale * (x*z - y*w), scale * (y*z + x*w), 1 - scale * (x*x + y*y)],
    ])
    out = torch.eye(4)
    out[:3, :3] = rotation
    return out


def navi_pose(annotation: dict) -> torch.Tensor:
    out = quaternion_matrix(annotation["camera"]["q"])
    out[:3, 3] = torch.tensor(annotation["camera"]["t"], dtype=torch.float32) / 1000.0
    return out


def angle_degrees(rotation: torch.Tensor) -> float:
    value = ((rotation.trace() - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(value.acos()).item()


def square_object_crop(image: Image.Image, depth: torch.Tensor) -> tuple[Image.Image, torch.Tensor, tuple[int, int]]:
    positions = (depth > 0).nonzero()
    if len(positions) == 0:
        raise ValueError("NAVI depth has no object pixels")
    top, left = positions.min(dim=0).values.tolist()
    bottom, right = (positions.max(dim=0).values + 1).tolist()
    side = max(bottom - top, right - left)
    centre_y, centre_x = (top + bottom) // 2, (left + right) // 2
    top = max(0, min(centre_y - side // 2, depth.shape[0] - side))
    left = max(0, min(centre_x - side // 2, depth.shape[1] - side))
    return (
        image.crop((left, top, left + side, top + side)),
        depth[top:top + side, left:left + side],
        (left, top),
    )


class ScanNetPairs:
    name = "scannet"

    def __init__(self, root: Path, resolution: int):
        self.root, self.resolution = root, resolution
        intrinsics = dict(np.load(root / "intrinsics.npz"))
        names = np.load(root / "test.npz")["name"]
        self.rows = []
        for room, sequence, first, second in names:
            scene = f"scene{room:04d}_{sequence:02d}"
            self.rows.append((scene, int(first), int(second), torch.tensor(intrinsics[scene]).float()))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        scene, first, second, matrix = self.rows[index]
        images, depths, frames = [], [], []
        for frame_id in (first, second):
            depth = read_depth(self.root / scene / "depth" / f"{frame_id}.png")
            with Image.open(self.root / scene / "color" / f"{frame_id}.jpg") as raw:
                aligned = align_rgb_to_depth(ImageOps.exif_transpose(raw).convert("RGB"), depth)
                image, frame = canonical(aligned, self.resolution)
            depth = frame.target(depth)
            images.append(image)
            depths.append(depth)
            frames.append(frame)
        pose_0 = torch.tensor(np.loadtxt(self.root / scene / "pose" / f"{first}.txt")).float()
        pose_1 = torch.tensor(np.loadtxt(self.root / scene / "pose" / f"{second}.txt")).float()
        target_from_source = pose_1.inverse() @ pose_0
        return {
            "id": f"{scene}-{first}-{second}",
            "images": images,
            "depths": depths,
            "intrinsics": [frames[0].intrinsics(matrix), frames[1].intrinsics(matrix)],
            "transform": target_from_source,
            "angle": angle_degrees(target_from_source[:3, :3]),
        }


class NAVIPairs:
    name = "navi"

    def __init__(self, root: Path, resolution: int):
        self.root, self.resolution = root, resolution
        rows = []
        generator = torch.Generator().manual_seed(8)
        for object_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            wild = object_dir / "wild_set"
            if not wild.exists() or not any(object_dir.glob("multiview_*")):
                continue
            annotations = json.loads((wild / "annotations.json").read_text())
            annotations = {row["filename"].split(".")[0]: row for row in annotations}
            ids = sorted(
                path.stem for path in (wild / "images").glob("*.jpg")
                if "_" not in path.stem
            )
            rotations = torch.stack([navi_pose(annotations[image_id])[:3, :3] for image_id in ids])
            for position, image_id in enumerate(ids):
                relative = rotations[position][None] @ rotations.transpose(1, 2)
                trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
                angle = torch.rad2deg((0.5 * trace - 0.5).clamp(-1, 1).acos())
                weight = ((angle > 0) & (angle <= 120)).float()
                if weight.sum() == 0:
                    continue
                pair = torch.multinomial(weight, 1, generator=generator).item()
                rows.append((object_dir.name, image_id, ids[pair], annotations[image_id], annotations[ids[pair]]))
        self.rows = rows[::4]

    def __len__(self):
        return len(self.rows)

    def _view(self, object_name: str, image_id: str, annotation: dict):
        wild = self.root / object_name / "wild_set"
        with Image.open(wild / "images" / f"{image_id}.jpg") as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB").copy()
        depth = navi_depth(wild / "depth" / f"{image_id}.png")
        image, depth, (crop_left, crop_top) = square_object_crop(image, depth)
        canonical_image, frame = canonical(image, self.resolution)
        focal = float(annotation["camera"]["focal_length"])
        matrix = torch.tensor([
            [focal, 0.0, annotation["image_size"][1] / 2],
            [0.0, focal, annotation["image_size"][0] / 2],
            [0.0, 0.0, 1.0],
        ])
        matrix[0, 2] -= crop_left
        matrix[1, 2] -= crop_top
        return canonical_image, frame.target(depth), frame.intrinsics(matrix), navi_pose(annotation)

    def __getitem__(self, index):
        object_name, first, second, annotation_0, annotation_1 = self.rows[index]
        view_0 = self._view(object_name, first, annotation_0)
        view_1 = self._view(object_name, second, annotation_1)
        transform = view_1[3] @ view_0[3].inverse()
        return {
            "id": f"{object_name}-{first}-{second}",
            "images": [view_0[0], view_1[0]],
            "depths": [view_0[1], view_1[1]],
            "intrinsics": [view_0[2], view_1[2]],
            "transform": transform,
            "angle": angle_degrees(transform[:3, :3]),
        }


class SPairPairs:
    name = "spair"

    def __init__(self, root: Path, resolution: int):
        self.root, self.resolution = root, resolution
        annotations = [json.loads(path.read_text()) for path in sorted((root / "PairAnnotation" / "test").glob("*.json"))]
        grouped = defaultdict(list)
        for row in annotations:
            grouped[row["category"]].append(row)
        self.rows = []
        for category in sorted(grouped):
            rows = grouped[category]
            random.Random(20).shuffle(rows)
            self.rows.extend(rows[:200])
        self.image_annotations = {}

    def __len__(self):
        return len(self.rows)

    def _annotations(self, category: str) -> dict:
        if category not in self.image_annotations:
            rows = [json.loads(path.read_text()) for path in (self.root / "ImageAnnotation" / category).glob("*.json")]
            self.image_annotations[category] = {row["filename"].split(".")[0]: row for row in rows}
        return self.image_annotations[category]

    def __getitem__(self, index):
        pair = self.rows[index]
        category = pair["category"]
        _, first, second = pair["filename"].split(":")[0].split("-")
        frames, images = [], []
        for image_id in (first, second):
            with Image.open(self.root / "JPEGImages" / category / f"{image_id}.jpg") as raw:
                image, frame = canonical(raw, self.resolution)
            images.append(image)
            frames.append(frame)
        annotations = self._annotations(category)
        keys = sorted(set(annotations[first]["kps"]) | set(annotations[second]["kps"]), key=int)
        points, visibility = [], []
        for image_id, frame in zip((first, second), frames):
            raw = annotations[image_id]["kps"]
            values = torch.tensor([raw.get(key) or [0, 0] for key in keys], dtype=torch.float32)
            visible = torch.tensor([bool(raw.get(key)) for key in keys])
            transformed = frame.points(values)
            visible &= (transformed >= 0).all(dim=1) & (transformed < self.resolution).all(dim=1)
            points.append(transformed)
            visibility.append(visible)
        bbox = pair["trg_bndbox"]
        bbox_width = (bbox[2] - bbox[0]) * frames[1].scale_x
        bbox_height = (bbox[3] - bbox[1]) * frames[1].scale_y
        return {
            "id": Path(pair["filename"]).stem,
            "images": images,
            "keypoints": points,
            "valid": visibility[0] & visibility[1],
            "threshold_scale": max(bbox_width, bbox_height) / self.resolution,
            "class": category,
            "viewpoint": int(pair["viewpoint_variation"]),
        }


DATASETS = {"scannet": ScanNetPairs, "navi": NAVIPairs, "spair": SPairPairs}
BATCH_ONE = {"kimi_k26", "moonvit_v2"}


def extract_pair(adapter, images, model_name: str):
    preprocess = adapter.preprocess()
    if model_name in BATCH_ONE:
        sides = []
        for position, image in enumerate(images):
            inputs = adapter.collate([preprocess(image)])
            sides.append({
                (batch.stage, batch.layer_index): batch.tokens[0]
                for batch in adapter.extract(inputs, [str(position)])
                if batch.stage != "tower" or batch.layer_index == adapter.num_layers
            })
        return {key: (sides[0][key], sides[1][key]) for key in sides[0]}
    inputs = adapter.collate([preprocess(image) for image in images])
    return {
        (batch.stage, batch.layer_index): (batch.tokens[0], batch.tokens[1])
        for batch in adapter.extract(inputs, ["0", "1"])
        if batch.stage != "tower" or batch.layer_index == adapter.num_layers
    }


def score_pair(dataset_name: str, instance: dict, pair_features: dict, args) -> dict:
    output = {}
    for (stage, layer), (tokens_0, tokens_1) in pair_features.items():
        features_0 = correspondence.dense_map(tokens_0).to(args.device)
        features_1 = correspondence.dense_map(tokens_1).to(args.device)
        if dataset_name == "spair":
            errors = correspondence.semantic_errors(
                features_0,
                features_1,
                instance["keypoints"][0],
                instance["keypoints"][1],
                instance["valid"],
                instance["threshold_scale"],
                resolution=args.resolution,
            )
            metrics = {"pck_0.1": correspondence.recall(errors, 0.1), "keypoints": len(errors)}
        else:
            error_3d, error_2d = correspondence.geometric_errors(
                features_0,
                features_1,
                instance["depths"][0],
                instance["depths"][1],
                instance["intrinsics"][0],
                instance["intrinsics"][1],
                instance["transform"],
                resolution=args.resolution,
                evaluation_side=args.evaluation_side,
                num_correspondences=args.num_correspondences,
            )
            metrics = {
                "recall_1cm": correspondence.recall(error_3d, 0.01),
                "recall_2cm": correspondence.recall(error_3d, 0.02),
                "recall_5cm": correspondence.recall(error_3d, 0.05),
                "recall_5px": correspondence.recall(error_2d, 5),
                "recall_10px": correspondence.recall(error_2d, 10),
                "recall_20px": correspondence.recall(error_2d, 20),
                "recall_25px": correspondence.recall(error_2d, 25),
                "recall_50px": correspondence.recall(error_2d, 50),
                "matches": len(error_3d),
            }
        output[f"{stage}:{layer}"] = metrics
    return output


def finite_mean(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor[torch.isfinite(tensor)].mean().item()


def summarise(dataset_name: str, rows: list[dict]) -> list[dict]:
    cells = []
    for cell in rows[0]["cells"]:
        metrics = rows[0]["cells"][cell]
        aggregate = {
            key: finite_mean([row["cells"][cell][key] for row in rows])
            for key, value in metrics.items() if isinstance(value, float)
        }
        stage, layer = cell.split(":")
        result = {"stage": stage, "layer_index": int(layer), **aggregate}
        if dataset_name == "spair":
            by_class = {}
            for category in sorted({row["class"] for row in rows}):
                by_class[category] = finite_mean([
                    row["cells"][cell]["pck_0.1"] for row in rows if row["class"] == category
                ])
            result["macro_pck_0.1"] = finite_mean(list(by_class.values()))
            result["by_class"] = by_class
        cells.append(result)
    return cells


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run training-free Stage correspondence.")
    ap.add_argument("--model", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--evaluation-side", type=int, default=64)
    ap.add_argument("--num-correspondences", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    return ap


def main() -> None:
    args = parser().parse_args()
    dataset = DATASETS[args.dataset](args.root, args.resolution)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    adapter = ADAPTERS[args.model](resolution=args.resolution, device=args.device)
    rows, started = [], time.perf_counter()
    for index in range(count):
        instance = dataset[index]
        cells = score_pair(args.dataset, instance, extract_pair(adapter, instance["images"], args.model), args)
        row = {"pair_id": instance["id"], "cells": cells}
        for key in ("angle", "class", "viewpoint"):
            if key in instance:
                row[key] = instance[key]
        rows.append(row)
        if (index + 1) % 10 == 0 or index + 1 == count:
            elapsed = time.perf_counter() - started
            print(f"{index + 1}/{count} pairs  {(index + 1) / elapsed:.2f} pair/s", flush=True)

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "model_id": adapter.model_id,
        "resolution": args.resolution,
        "evaluation_side": args.evaluation_side,
        "num_correspondences": args.num_correspondences,
        "pairs": count,
        "seconds": round(time.perf_counter() - started, 1),
        "cells": summarise(args.dataset, rows),
        "measurements": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "measurements"}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
