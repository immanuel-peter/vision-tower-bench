"""Training-free correspondence scoring on frozen Stage features.

The matching rule follows Probe3D: cosine nearest neighbours, Lowe's ratio test for
geometric correspondence, and direct nearest-neighbour lookup for SPair keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Matches:
    source: torch.Tensor
    target: torch.Tensor
    confidence: torch.Tensor


def dense_map(tokens: torch.Tensor) -> torch.Tensor:
    """Convert one image's ``(tokens, width)`` patch sequence to ``(width, h, w)``."""
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be (token, width), got {tuple(tokens.shape)}")
    side = isqrt(tokens.shape[0])
    if side * side != tokens.shape[0]:
        raise ValueError(f"{tokens.shape[0]} tokens do not form a square grid")
    return tokens.transpose(0, 1).reshape(tokens.shape[1], side, side).contiguous()


def resize_features(features: torch.Tensor, side: int) -> torch.Tensor:
    """Interpolate a Stage map onto a shared square evaluation grid."""
    if features.shape[-2:] == (side, side):
        return features.float()
    return F.interpolate(
        features[None].float(), (side, side), mode="bicubic", align_corners=False
    )[0]


def grid_centres(side: int, resolution: int, *, device=None) -> torch.Tensor:
    """Return ``(x, y)`` pixel centres for a square feature grid."""
    axis = (torch.arange(side, device=device, dtype=torch.float32) + 0.5) * (
        resolution / side
    )
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def sample_map(values: torch.Tensor, xy: torch.Tensor, resolution: int) -> torch.Tensor:
    """Bilinearly sample a ``(channels, h, w)`` map at output-image pixel positions."""
    if values.ndim == 2:
        values = values[None]
    # Stage features may be bfloat16, but grid_sample requires the sampling grid
    # and input to have the same floating-point dtype.
    grid = xy.float().clone()
    grid[:, 0] = 2 * grid[:, 0] / resolution - 1
    grid[:, 1] = 2 * grid[:, 1] / resolution - 1
    sampled = F.grid_sample(
        values[None].float(), grid[None, None], mode="bilinear", align_corners=False
    )
    return sampled[0, :, 0].transpose(0, 1)


def ratio_matches(
    source: torch.Tensor,
    target: torch.Tensor,
    num_correspondences: int,
    *,
    chunk_size: int = 256,
) -> Matches:
    """Return the most distinctive cosine matches under Lowe's ratio test.

    ``source`` and ``target`` are point-major matrices. The chunked matrix multiply keeps
    wide Projector features from materialising a large query-by-target-by-width tensor.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be (points, shared_width) matrices")
    if len(source) == 0 or len(target) < 2:
        raise ValueError("matching needs a source point and at least two target points")

    source = F.normalize(source.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    neighbours, ratios = [], []
    for start in range(0, len(source), chunk_size):
        similarity = source[start:start + chunk_size] @ target.transpose(0, 1)
        top = similarity.topk(2, dim=1)
        distance = (1 - top.values).clamp_min(1e-9)
        neighbours.append(top.indices[:, 0])
        ratios.append(1 - distance[:, 0] / distance[:, 1])
    neighbour = torch.cat(neighbours)
    confidence = torch.cat(ratios)
    count = min(num_correspondences, len(source))
    chosen = confidence.topk(count).indices
    return Matches(chosen, neighbour[chosen], confidence[chosen])


def _valid_points(
    features: torch.Tensor,
    depth: torch.Tensor,
    side: int,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = resize_features(features, side).flatten(1).transpose(0, 1)
    xy = grid_centres(side, resolution, device=features.device)
    sampled_depth = sample_map(depth.to(features.device), xy, resolution)[:, 0]
    valid = torch.isfinite(sampled_depth) & (sampled_depth > 0)
    return features[valid], xy[valid], sampled_depth[valid]


def unproject(xy: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((xy, torch.ones_like(xy[:, :1])), dim=1)
    rays = homogeneous @ intrinsics.to(xy).inverse().transpose(0, 1)
    return rays * depth[:, None]


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    return (homogeneous @ transform.to(points).transpose(0, 1))[:, :3]


def project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    uvd = points @ intrinsics.to(points).transpose(0, 1)
    return uvd[:, :2] / uvd[:, 2:].clamp_min(1e-9)


def geometric_errors(
    features_0: torch.Tensor,
    features_1: torch.Tensor,
    depth_0: torch.Tensor,
    depth_1: torch.Tensor,
    intrinsics_0: torch.Tensor,
    intrinsics_1: torch.Tensor,
    target_from_source: torch.Tensor,
    *,
    resolution: int = 448,
    evaluation_side: int = 32,
    num_correspondences: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 3D metric and target-image projection errors for one geometric pair."""
    f0, xy0, d0 = _valid_points(features_0, depth_0, evaluation_side, resolution)
    f1, xy1, d1 = _valid_points(features_1, depth_1, evaluation_side, resolution)
    matches = ratio_matches(f0, f1, num_correspondences)

    xyz0 = unproject(xy0, d0, intrinsics_0)[matches.source]
    xyz1 = unproject(xy1, d1, intrinsics_1)[matches.target]
    xyz0_in_1 = transform_points(xyz0, target_from_source)
    error_3d = (xyz0_in_1 - xyz1).norm(dim=1)
    error_2d = (project(xyz0_in_1, intrinsics_1) - xy1[matches.target]).norm(dim=1)
    return error_3d, error_2d


def semantic_errors(
    features_0: torch.Tensor,
    features_1: torch.Tensor,
    source_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    valid: torch.Tensor,
    threshold_scale: float,
    *,
    resolution: int = 448,
) -> torch.Tensor:
    """Return SPair keypoint errors normalised by target bounding-box scale."""
    source_keypoints = source_keypoints[valid]
    target_keypoints = target_keypoints[valid]
    if len(source_keypoints) == 0:
        return torch.empty(0)

    side = features_1.shape[-1]
    query = sample_map(features_0, source_keypoints.to(features_0), resolution)
    target = features_1.flatten(1).transpose(0, 1)
    query = F.normalize(query.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    predicted = (query @ target.transpose(0, 1)).argmax(dim=1)
    centres = grid_centres(side, resolution, device=features_1.device)
    error = (centres[predicted] - target_keypoints.to(centres)).norm(dim=1)
    return error / (resolution * threshold_scale)


def recall(errors: torch.Tensor, threshold: float) -> float:
    if errors.numel() == 0:
        return float("nan")
    return (errors < threshold).float().mean().item()


def paired_bootstrap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, float | int | str]:
    """Bootstrap a paired increasing metric. Positive values favour ``first``."""
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired measurements must have the same one-dimensional shape")
    if not len(first):
        raise ValueError("cannot bootstrap empty measurements")
    difference = first.float() - second.float()
    generator = torch.Generator().manual_seed(seed)
    draws = []
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        index = torch.randint(len(difference), (count, len(difference)), generator=generator)
        draws.append(difference[index].mean(dim=1))
    distribution = torch.cat(draws)
    tail = (1 - confidence) / 2
    lower, upper = torch.quantile(distribution, torch.tensor([tail, 1 - tail]))
    return {
        "point_estimate": difference.mean().item(),
        "confidence": confidence,
        "lower": lower.item(),
        "upper": upper.item(),
        "resamples": resamples,
        "bootstrap_seed": seed,
        "difference": "first_minus_second",
    }
