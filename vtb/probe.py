"""The shared semantic readout.

One attention pool over the cached token grid, then a linear classifier. The Tower stays
frozen; the accuracy of this head is the measurement. Identical for every model, Stage,
and Relative Depth point, which is what makes cells comparable.
"""

from dataclasses import dataclass

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Collapse a token grid to one vector with a learned query, then classify.

    A single query rather than a transformer block, so the head stays near the one to two
    million parameters the protocol allows and cannot itself learn the task.
    """

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
    """The ablation column: average the tokens instead of attending over them."""

    def __init__(self, dim: int, num_classes: int, width: int = 512):
        super().__init__()
        self.head = nn.Linear(dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens.mean(dim=1))


@dataclass(frozen=True)
class Reducer:
    """A frozen linear map from token width to a common width.

    Trained parameter count scales with token width, so a `projected` cell at 7168 gets a
    bigger head than a `tower` cell at 1024 and part of any gap would be the head rather
    than the features. Fitting this reduction and freezing it makes the trained head
    identical everywhere. It is fitted with PCA rather than drawn at random, because a
    random map discards more from a wide Stage than a narrow one and would push the
    headline claim in the direction the project is trying to test.
    """

    basis: torch.Tensor
    mean: torch.Tensor

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens - self.mean) @ self.basis

    @classmethod
    def fit(cls, tokens: torch.Tensor, width: int, oversample: int = 16) -> "Reducer":
        flat = tokens.reshape(-1, tokens.shape[-1]).float()
        mean = flat.mean(dim=0, keepdim=True)
        # Randomized SVD for the leading components only. A full SVD of the widest cell
        # here is 145,600 by 7168 and solves for all 7168 directions to keep 512.
        _, _, v = torch.svd_lowrank(flat - mean, q=min(width + oversample, min(flat.shape)))
        return cls(basis=v[:, :width].contiguous(), mean=mean)


def build(kind: str, dim: int, num_classes: int, width: int = 512) -> nn.Module:
    return {"attention": AttentionPool, "mean": MeanPool}[kind](dim, num_classes, width)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
