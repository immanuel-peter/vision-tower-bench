"""Compute complete per-tensor and concatenated per-block Kimi weight drift."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def group_name(filename, name):
    if filename == "projector.safetensors":
        return "projector"
    block = re.match(r"encoder\.blocks\.(\d+)\.", name)
    if block:
        return f"block_{int(block[1]) + 1:02d}"
    if "pos_emb" in name:
        return "spatial_position_embedding"
    if name.startswith("patch_embed"):
        return "patch_stem"
    return "final_norm"


def metrics(sums):
    return {"cosine": sums["dot"] / math.sqrt(sums["first_squared_norm"] * sums["second_squared_norm"]),
            "relative_l2": math.sqrt(sums["difference_squared_norm"] / sums["first_squared_norm"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tensors, groups, hashes = [], {}, {}
    for filename in ("model.safetensors", "projector.safetensors"):
        first, second = load_file(args.first / filename), load_file(args.second / filename)
        if first.keys() != second.keys():
            raise ValueError("sibling checkpoints have different tensor names")
        for source in (args.first, args.second):
            with (source / filename).open("rb") as stream:
                hashes[str(source / filename)] = hashlib.file_digest(stream, "sha256").hexdigest()
        for name in first:
            if first[name].shape != second[name].shape or first[name].dtype != second[name].dtype:
                raise ValueError("sibling checkpoint tensor shapes/dtypes differ")
            x, y = first[name].flatten().double(), second[name].flatten().double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("nonfinite source weight")
            sums = dict(dot=float(x @ y), first_squared_norm=float(x @ x),
                        second_squared_norm=float(y @ y), difference_squared_norm=float(((y - x) ** 2).sum()))
            group = group_name(filename, name)
            tensors.append(dict(file=filename, name=name, group=group, shape=list(first[name].shape),
                                elements=x.numel(), bit_equal=torch.equal(first[name].view(torch.uint8), second[name].view(torch.uint8)),
                                **sums, **metrics(sums)))
            aggregate = groups.setdefault(group, dict(dot=0., first_squared_norm=0., second_squared_norm=0.,
                                                      difference_squared_norm=0., elements=0, tensors=0))
            for key, value in sums.items():
                aggregate[key] += value
            aggregate["elements"] += x.numel()
            aggregate["tensors"] += 1
    for group in groups.values():
        group.update(metrics(group))
    report = dict(first_source=str(args.first), second_source=str(args.second), source_file_sha256=hashes,
                  source_manifests={role: json.loads((path / "export_manifest.json").read_text())
                                    for role, path in (("first", args.first), ("second", args.second))
                                    if (path / "export_manifest.json").is_file()},
                  method="FP64 dot products and squared norms; group metrics equal concatenating every parameter in that group",
                  tensor_count=len(tensors), changed_tensors=sum(not row["bit_equal"] for row in tensors),
                  tensors=tensors, groups=groups,
                  temporal_position_buffer="Derived sincos, same pinned configuration; not saved as a tensor, not added for T=1")
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
