"""Opt-in channel-standardized PCA for the recovered figure diagnostic."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StandardizedReducer:
    basis: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return ((tokens - self.mean) / self.scale) @ self.basis

    @classmethod
    def fit(cls, tokens: torch.Tensor, width: int, oversample: int = 16):
        flat = tokens.reshape(-1, tokens.shape[-1]).float()
        mean = flat.mean(dim=0, keepdim=True)
        scale = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
        _, _, basis = torch.svd_lowrank(
            (flat - mean) / scale, q=min(width + oversample, min(flat.shape)),
        )
        return cls(basis[:, :width].contiguous(), mean, scale)
