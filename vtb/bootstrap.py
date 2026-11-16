"""Paired bootstrap intervals over shared images or image pairs."""

import torch


def bootstrap_mean(
    differences: torch.Tensor,
    *,
    resamples: int,
    confidence: float,
    seed: int,
    batch_size: int,
) -> dict[str, float | int]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        indices = torch.randint(len(differences), (count, len(differences)), generator=generator)
        samples.append(differences[indices].mean(dim=1))
    distribution = torch.cat(samples)
    tail = (1.0 - confidence) / 2.0
    lower, upper = torch.quantile(distribution, torch.tensor([tail, 1.0 - tail]))
    return {
        "point_estimate": differences.mean().item(),
        "confidence": confidence,
        "lower": lower.item(),
        "upper": upper.item(),
        "resamples": resamples,
        "bootstrap_seed": seed,
    }


def paired_bootstrap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, float | int | str]:
    """Paired bootstrap of first-minus-second over one measurement per image pair."""
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired measurements must have the same one-dimensional shape")
    if not len(first):
        raise ValueError("cannot bootstrap empty measurements")
    difference = first.float() - second.float()
    interval = bootstrap_mean(difference, resamples=resamples, confidence=confidence, seed=seed, batch_size=batch_size)
    return interval | {"difference": "first_minus_second"}


def seed_mean_differences(first: torch.Tensor, second: torch.Tensor, kind: str) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(f"{kind} tensors must have the same (seeds, images) shape")
    if first.shape[1] == 0:
        raise ValueError("cannot bootstrap an empty test split")
    return first.float().mean(dim=0) - second.float().mean(dim=0)


def paired_image_bootstrap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, float | int]:
    """Paired bootstrap of first-minus-second accuracy over shared test images."""
    differences = seed_mean_differences(first, second, "correctness")
    return bootstrap_mean(differences, resamples=resamples, confidence=confidence, seed=seed, batch_size=batch_size)


def paired_metric_bootstrap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    higher_is_better: bool,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, float | int | str]:
    """Paired bootstrap of first-minus-second (or the reverse if lower is better)."""
    differences = seed_mean_differences(first, second, "metric")
    direction = "first_minus_second"
    if not higher_is_better:
        differences = -differences
        direction = "second_minus_first"
    interval = bootstrap_mean(differences, resamples=resamples, confidence=confidence, seed=seed, batch_size=batch_size)
    return interval | {"difference": direction}
