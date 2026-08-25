from dataclasses import dataclass

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, dim: int, num_classes: int, width: int = 512, heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, width) * width**-0.5)
        self.to_kv = nn.Linear(dim, 2 * width, bias=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        keys, values = self.to_kv(tokens).chunk(2, dim=-1)
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, keys, values, need_weights=False)
        return self.head(self.norm(pooled.squeeze(1)))


class MeanPool(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens.mean(dim=1))


@dataclass(frozen=True)
class Reducer:
    """Frozen PCA map fit on training features for capacity matching."""

    basis: torch.Tensor
    mean: torch.Tensor

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens - self.mean) @ self.basis

    @classmethod
    def fit(cls, tokens: torch.Tensor, width: int, oversample: int = 16) -> "Reducer":
        flat = tokens.reshape(-1, tokens.shape[-1]).float()
        mean = flat.mean(dim=0, keepdim=True)
        # Randomized SVD avoids solving 7,168 directions in the largest 145,600 by 7,168 cell.
        _, _, v = torch.svd_lowrank(flat - mean, q=min(width + oversample, min(flat.shape)))
        return cls(basis=v[:, :width].contiguous(), mean=mean)


def build(kind: str, dim: int, num_classes: int, width: int = 512) -> nn.Module:
    if kind == "attention":
        return AttentionPool(dim, num_classes, width)
    if kind == "mean":
        return MeanPool(dim, num_classes)
    raise ValueError(f"unknown readout: {kind}")


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
