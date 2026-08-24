from dataclasses import dataclass, replace
from math import isqrt

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

STAGES = ("tower", "merged", "projected")


def concat(batches: list["FeatureBatch"]) -> "FeatureBatch":
    """Join batches that share a model, Stage, and depth point into one."""
    head = batches[0]
    return replace(
        head,
        tokens=torch.cat([b.tokens for b in batches]),
        image_ids=[image_id for b in batches for image_id in b.image_ids],
    )


@dataclass(frozen=True)
class FeatureBatch:
    """Patch tokens from one model at one Stage and one Relative Depth point.

    Every adapter returns these and every probe consumes them, so the field set
    is the contract that keeps models comparable. No CLS token: none of the four
    multimodal Towers has one, so tokens is patch-only everywhere.
    """

    tokens: torch.Tensor
    image_ids: list[str]
    model_id: str
    stage: str
    layer_index: int
    num_layers: int
    resolution: int
    pooled_to: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage {self.stage!r} not in {STAGES}")
        if self.tokens.ndim != 3:
            raise ValueError(f"tokens must be (batch, token, dim), got {tuple(self.tokens.shape)}")
        if self.tokens.shape[0] != len(self.image_ids):
            raise ValueError(f"{self.tokens.shape[0]} rows but {len(self.image_ids)} image ids")

    @property
    def relative_depth(self) -> float:
        return self.layer_index / self.num_layers

    def pooled(self, side: int) -> "FeatureBatch":
        """Average the patch grid down to side x side tokens.

        The semantic pillar caches this instead of every patch token (ADR-0005).
        It lives here so that every adapter and every Stage pools the same way,
        which is what keeps models comparable.
        """
        count = self.tokens.shape[1]
        grid = isqrt(count)
        if grid * grid != count:
            raise ValueError(f"{count} tokens do not form a square grid")
        if side > grid:
            raise ValueError(f"cannot pool a {grid}x{grid} grid up to {side}x{side}")

        rows, _, dim = self.tokens.shape
        spatial = self.tokens.transpose(1, 2).reshape(rows, dim, grid, grid)
        # Average in fp32 so the mean does not lose precision at bf16.
        small = F.adaptive_avg_pool2d(spatial.float(), side).to(self.tokens.dtype)
        return replace(self, tokens=small.flatten(2).transpose(1, 2), pooled_to=side)

    @property
    def nbytes(self) -> int:
        return self.tokens.nbytes

    def metadata(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "stage": self.stage,
            "layer_index": str(self.layer_index),
            "num_layers": str(self.num_layers),
            "relative_depth": f"{self.relative_depth:.4f}",
            "resolution": str(self.resolution),
            "pooled_to": str(self.pooled_to) if self.pooled_to else "none",
            "image_ids": "\n".join(self.image_ids),
        }

    def save(self, path) -> None:
        save_file({"tokens": self.tokens.contiguous()}, str(path), metadata=self.metadata())
