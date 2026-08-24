"""Reads feature shards back out of the cache.

`extract.py` writes one shard per Stage, Relative Depth point, and batch. Probes want
one tensor per (Stage, depth point), so this reassembles them in shard order.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

SHARD = re.compile(r"^(?P<stage>[a-z]+)_L(?P<layer>\d+)_(?P<shard>\d+)\.safetensors$")


def slices(run_dir: Path) -> list[tuple[str, int]]:
    """Every (stage, layer_index) the run wrote, in Stage then depth order."""
    found = {(m["stage"], int(m["layer"])) for p in Path(run_dir).iterdir() if (m := SHARD.match(p.name))}
    return sorted(found, key=lambda s: (s[0], s[1]))


def shards(run_dir: Path, stage: str, layer: int) -> Iterator[Path]:
    pattern = f"{stage}_L{layer:02d}_*.safetensors"
    return iter(sorted(Path(run_dir).glob(pattern)))


def load(run_dir: Path, stage: str, layer: int) -> tuple[torch.Tensor, list[str], dict[str, str]]:
    """Concatenate one Stage at one depth point into (images, tokens, dim)."""
    parts, image_ids, meta = [], [], {}
    for path in shards(run_dir, stage, layer):
        with safe_open(path, framework="pt") as handle:
            meta = handle.metadata()
            parts.append(handle.get_tensor("tokens"))
            image_ids.extend(meta["image_ids"].split("\n"))
    if not parts:
        raise FileNotFoundError(f"no shards for {stage} L{layer:02d} under {run_dir}")
    return torch.cat(parts), image_ids, meta
