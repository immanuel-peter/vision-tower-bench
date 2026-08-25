"""Dense readouts for the geometry pillar.

Ported from Probe3D (mbanani/probe3d, CVPR 2024) so the decoder matches the published
protocol. The heads read a list of feature maps, one per Relative Depth point, which is
why the geometry cache keeps full patch tokens instead of the 4x4 grid the semantic
pillar uses (ADR-0005).
"""

from math import isqrt

import torch
from torch import nn
from torch.nn.functional import interpolate

from vtb.feature_batch import FeatureBatch

PROBE3D_COMMIT = "https://github.com/mbanani/probe3d"


def dense_map(batch: FeatureBatch) -> torch.Tensor:
    """Lay a batch's patch tokens back onto their grid as (images, dim, height, width).

    Probe3D calls this the `dense` output type. Its backbones drop the CLS and register
    tokens first; ours are patch-only already.
    """
    if batch.pooled_to is not None:
        raise ValueError("geometry needs full patch tokens, not a pooled grid (ADR-0005)")
    count = batch.tokens.shape[1]
    side = isqrt(count)
    if side * side != count:
        raise ValueError(f"{count} tokens do not form a square grid")
    rows, _, dim = batch.tokens.shape
    return batch.tokens.transpose(1, 2).reshape(rows, dim, side, side).contiguous()


def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    if num_layers == 1:
        return nn.Conv2d(input_dim, output_dim, kernel_size)
    modules = [nn.Conv2d(input_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
    for _ in range(num_layers - 2):
        modules += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
    modules.append(nn.Conv2d(hidden_dim, output_dim, kernel_size))
    return nn.Sequential(*modules)


class Linear(nn.Module):
    """The cheapest Probe3D head. One convolution over upsampled features."""

    def __init__(self, input_dims, output_dim: int, kernel_size: int = 1):
        super().__init__()
        width = input_dims if isinstance(input_dims, int) else sum(input_dims)
        self.conv = nn.Conv2d(width, output_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        joined = torch.cat(feats, dim=1) if isinstance(feats, list) else feats
        return self.conv(interpolate(joined, scale_factor=4, mode="bilinear", align_corners=True))


class MultiscaleHead(nn.Module):
    """Projects each feature map, joins them at the finest grid, then upsamples 8x."""

    def __init__(self, input_dims: list[int], output_dim: int, hidden_dim: int = 512, kernel_size: int = 1):
        super().__init__()
        self.convs = nn.ModuleList(
            [make_conv(dim, None, hidden_dim, 1, kernel_size) for dim in input_dims]
        )
        self.conv_mid = make_conv(len(input_dims) * hidden_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        projected = [conv(feat) for conv, feat in zip(self.convs, feats)]
        height, width = projected[-1].shape[-2:]
        projected = [
            interpolate(feat, (height, width), mode="bilinear", align_corners=True)
            for feat in projected
        ]
        joined = torch.cat(projected, dim=1).relu()
        joined = interpolate(joined, scale_factor=2, mode="bilinear", align_corners=True)
        joined = self.conv_mid(joined).relu()
        joined = interpolate(joined, scale_factor=4, mode="bilinear", align_corners=True)
        return self.conv_out(joined)


class DepthBins(nn.Module):
    """Turns per-bin scores into a depth by taking their expected value, as AdaBins does."""

    def __init__(self, min_depth: float = 0.001, max_depth: float = 10.0, n_bins: int = 256):
        super().__init__()
        self.min_depth, self.max_depth, self.n_bins = min_depth, max_depth, n_bins

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=scores.device)
        weights = scores.relu() + 0.1
        weights = weights / weights.sum(dim=1, keepdim=True)
        return torch.einsum("ikhw,k->ihw", weights, bins).unsqueeze(1)


class DepthHead(nn.Module):
    def __init__(self, input_dims: list[int], head: str = "multiscale", hidden_dim: int = 512):
        super().__init__()
        self.predict = DepthBins()
        self.head = _build(head, input_dims, self.predict.n_bins, hidden_dim)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        return self.predict(self.head(feats))


class SurfaceNormalHead(nn.Module):
    def __init__(self, input_dims: list[int], head: str = "multiscale", hidden_dim: int = 512):
        super().__init__()
        self.head = _build(head, input_dims, 3, hidden_dim)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        return self.head(feats)


def _build(head: str, input_dims: list[int], output_dim: int, hidden_dim: int) -> nn.Module:
    if head == "linear":
        return Linear(input_dims, output_dim)
    if head == "multiscale":
        return MultiscaleHead(input_dims, output_dim, hidden_dim)
    raise ValueError(f"unknown head {head!r}")


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
