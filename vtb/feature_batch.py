from dataclasses import dataclass

import torch
from safetensors.torch import save_file

STAGES = ("tower", "merged", "projected")


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
            "image_ids": "\n".join(self.image_ids),
        }

    def save(self, path) -> None:
        save_file({"tokens": self.tokens.contiguous()}, str(path), metadata=self.metadata())
