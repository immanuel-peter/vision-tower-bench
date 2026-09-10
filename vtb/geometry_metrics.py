"""Probe3D depth and surface-normal metrics. Functions return per-image tensors."""

import torch

DEPTH_THRESHOLDS = (1.25, 1.25**2, 1.25**3)
NORMAL_THRESHOLDS = (11.25, 22.5, 30.0)
# Standard NYU depth crop for 480x640 images.
NYU_CROP = (slice(45, 471), slice(41, 601))


def match_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = (target > 0).float()
    flat_pr = prediction.flatten(1)
    flat_gt = target.flatten(1)
    flat_valid = valid.flatten(1)

    count = flat_valid.sum(dim=1).clamp(min=1)
    mean_pr = (flat_pr * flat_valid).sum(dim=1) / count
    mean_gt = (flat_gt * flat_valid).sum(dim=1) / count
    centered_pr = (flat_pr - mean_pr[:, None]) * flat_valid
    centered_gt = (flat_gt - mean_gt[:, None]) * flat_valid

    scale = (centered_pr * centered_gt).sum(dim=1) / centered_pr.pow(2).sum(dim=1).clamp(min=1e-9)
    shift = mean_gt - scale * mean_pr
    return (prediction * scale[:, None, None] + shift[:, None, None]).reshape(target.shape)


def evaluate_depth(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale_invariant: bool = False,
    nyu_crop: bool = False,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"{tuple(prediction.shape)} != {tuple(target.shape)}")
    if prediction.ndim == 4:
        prediction, target = prediction.squeeze(1), target.squeeze(1)
    if nyu_crop:
        rows, cols = NYU_CROP
        prediction, target = prediction[..., rows, cols], target[..., rows, cols]
    if scale_invariant:
        prediction = match_scale_and_shift(prediction, target)

    valid = (target > 0).detach().float()
    prediction = prediction * valid
    count = valid.sum(dim=(1, 2)).clamp(min=1)

    ratio = torch.maximum(
        target / prediction.clamp(min=1e-9), prediction / target.clamp(min=1e-9)
    )
    metrics = {
        f"d{i + 1}": ((ratio < t).float() * valid).sum(dim=(1, 2)) / count
        for i, t in enumerate(DEPTH_THRESHOLDS)
    }
    metrics["rmse"] = ((target - prediction).pow(2) * valid).sum(dim=(1, 2)).div(count).sqrt()
    return metrics


def evaluate_surface_normal(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> dict[str, torch.Tensor]:
    prediction = prediction[:, :3]
    if prediction.shape != target.shape:
        raise ValueError(f"{tuple(prediction.shape)} != {tuple(target.shape)}")

    cosine = torch.cosine_similarity(prediction, target, dim=1).clamp(-1.0, 1.0)
    error = torch.acos(cosine) * 180.0 / torch.pi
    valid = valid.squeeze(1).float()
    error = error * valid
    count = valid.sum(dim=(1, 2)).clamp(min=1)

    metrics = {
        f"d{i + 1}": ((error < t).float() * valid).sum(dim=(1, 2)) / count
        for i, t in enumerate(NORMAL_THRESHOLDS)
    }
    metrics["rmse"] = error.pow(2).sum(dim=(1, 2)).div(count).sqrt()
    metrics["mean_deg"] = error.sum(dim=(1, 2)) / count
    return metrics


def depth_si_loss(prediction, target, alpha: float = 10.0, lambda_scale: float = 0.85, eps: float = 1e-5):
    """Eigen log-space depth loss, Probe3D defaults (lambda 0.85, sqrt per image)."""
    valid = (target > 0).detach().float()
    count = valid.sum(dim=(-1, -2)).clamp(min=1)
    diff = (prediction.clamp(min=eps).log() - target.clamp(min=eps).log()) * valid
    mean = diff.pow(2).sum(dim=(-2, -1)) / count
    variance = diff.sum(dim=(-2, -1)).pow(2) / count.pow(2)
    return alpha * (mean - lambda_scale * variance).clamp(min=0).sqrt().mean()


def normal_loss(prediction, target, valid):
    cosine = torch.cosine_similarity(prediction[:, :3], target, dim=1).clamp(-1.0, 1.0)
    valid = valid.squeeze(1).float()
    count = valid.sum(dim=(1, 2)).clamp(min=1)
    return (((1.0 - cosine) * valid).sum(dim=(1, 2)) / count).mean()
