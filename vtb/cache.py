import re
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from vtb.feature_batch import FeatureBatch, concat

SHARD = re.compile(r"^(?P<stage>[a-z]+)_L(?P<layer>\d+)_(?P<shard>\d+)\.safetensors$")


def slices(run_dir: Path) -> list[tuple[str, int]]:
    found = {(m["stage"], int(m["layer"])) for p in Path(run_dir).iterdir() if (m := SHARD.match(p.name))}
    return sorted(found, key=lambda s: (s[0], s[1]))


def shards(run_dir: Path, stage: str, layer: int) -> Iterator[Path]:
    pattern = f"{stage}_L{layer:02d}_*.safetensors"
    return iter(sorted(Path(run_dir).glob(pattern)))


def load(run_dir: Path, stage: str, layer: int) -> tuple[torch.Tensor, list[str], dict[str, str]]:
    parts: list[torch.Tensor] = []
    image_ids: list[str] = []
    metadata: dict[str, str] = {}
    for path in shards(run_dir, stage, layer):
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            parts.append(handle.get_tensor("tokens"))
            image_ids.extend(metadata["image_ids"].split("\n"))
    if not parts:
        raise FileNotFoundError(f"no shards for {stage} L{layer:02d} under {run_dir}")
    return torch.cat(parts), image_ids, metadata


def load_batch(run_dir: Path, stage: str, layer: int) -> FeatureBatch:
    tokens, image_ids, meta = load(run_dir, stage, layer)
    pooled = meta.get("pooled_to", "none")
    return FeatureBatch(
        tokens=tokens,
        image_ids=image_ids,
        model_id=meta["model_id"],
        stage=stage,
        layer_index=layer,
        num_layers=int(meta["num_layers"]),
        resolution=int(meta["resolution"]),
        pooled_to=None if pooled == "none" else int(pooled),
    )


class ShardWriter:

    def __init__(self, run_dir: Path, images: int = 512):
        self.run_dir = Path(run_dir)
        self.images = images
        self.buffers: dict[tuple[str, int], list[FeatureBatch]] = {}
        self.counts: dict[tuple[str, int], int] = {}
        self.shard_counts: dict[tuple[str, int], int] = {}

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
        index = self.shard_counts.get(key, 0)
        concat(batches).save(self.run_dir / f"{stage}_L{layer:02d}_{index:05d}.safetensors")
        self.shard_counts[key] = index + 1
        self.counts[key] = 0

    def close(self) -> None:
        for key in list(self.buffers):
            self.flush(key)

    @property
    def slice_count(self) -> int:
        return len(self.shard_counts)
