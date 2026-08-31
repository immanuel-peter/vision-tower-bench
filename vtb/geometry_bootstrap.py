"""Paired image bootstrap utilities for geometry headline comparisons."""

import torch


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
    """Bootstrap the first cell's advantage over paired test images.

    Inputs contain one metric value per seed and image, shaped ``(seeds, images)``.
    Seed means are formed before resampling. The reported difference is positive when
    the first cell is better: first-minus-second for increasing metrics and
    second-minus-first for decreasing metrics.
    """
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("metric tensors must have the same (seeds, images) shape")
    if first.shape[1] == 0:
        raise ValueError("cannot bootstrap an empty test split")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    first_mean = first.float().mean(dim=0)
    second_mean = second.float().mean(dim=0)
    differences = first_mean - second_mean
    direction = "first_minus_second"
    if not higher_is_better:
        differences = -differences
        direction = "second_minus_first"

    generator = torch.Generator().manual_seed(seed)
    samples = []
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        indices = torch.randint(
            len(differences), (count, len(differences)), generator=generator
        )
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
        "difference": direction,
    }
