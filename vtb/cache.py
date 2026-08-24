"""Reads feature shards back out of the cache.

`extract.py` writes one shard per Stage, Relative Depth point, and batch. Probes want
one tensor per (Stage, depth point), so this reassembles them in shard order.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from vtb.feature_batch import FeatureBatch, concat

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


class ShardWriter:
    """Buffers batches per Stage and depth point, writing a shard every `images` images.

    Extraction batch size answers to GPU memory, and for a packed-sequence Tower to
    attention cost, which forces batch 1 (ADR-0009). Writing one file per batch would tie
    the file count to that choice: at batch 1 a 13,000 image run wrote 130,001 files, and
    the ImageNet-100 train split would write over a million for one model.
    """

    def __init__(self, run_dir: Path, images: int = 512):
        self.run_dir = Path(run_dir)
        self.images = images
        self.buffers: dict[tuple[str, int], list[FeatureBatch]] = {}
        self.counts: dict[tuple[str, int], int] = {}
        self.shards: dict[tuple[str, int], int] = {}

    def add(self, batch: FeatureBatch) -> None:
        key = (batch.stage, batch.layer_index)
        self.buffers.setdefault(key, []).append(batch)
        self.counts[key] = self.counts.get(key, 0) + batch.tokens.shape[0]
        if self.counts[key] >= self.images:
            self.flush(key)

    def flush(self, key: tuple[str, int]) -> None:
        batches = self.buffers.pop(key, None)
        if not batches:
            return
        stage, layer = key
        index = self.shards.get(key, 0)
        concat(batches).save(self.run_dir / f"{stage}_L{layer:02d}_{index:05d}.safetensors")
        self.shards[key] = index + 1
        self.counts[key] = 0

    def close(self) -> None:
        for key in list(self.buffers):
            self.flush(key)

    @property
    def slices(self) -> int:
        return len(self.shards)
