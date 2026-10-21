"""Count loss and per-image count errors on the declared density grid."""

import math

import torch

from vtb.counting import BINS, DENSITY_SCALE, count_bin, masked_count


def density_loss(prediction: torch.Tensor, target: torch.Tensor, scale: float = DENSITY_SCALE) -> torch.Tensor:
    """Pixel MSE on density, measured in scaled units so the loss is not vanishingly small."""
    if prediction.shape != target.shape:
        raise ValueError(f"{tuple(prediction.shape)} != {tuple(target.shape)}")
    return (prediction - target).mul(scale).pow(2).mean()


def evaluate_count(
    prediction: torch.Tensor, true: torch.Tensor, valid: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Per-image counts and errors against integer true counts. signed_error is predicted minus true."""
    predicted = masked_count(prediction.double(), valid)
    true = true.to(predicted)
    signed = predicted - true
    return {
        "abs_error": signed.abs(),
        "sq_error": signed.pow(2),
        "signed_error": signed,
        "predicted_count": predicted,
        "true_count": true,
    }


def summarise(metrics: dict[str, torch.Tensor], bins: list[str]) -> dict:
    """MAE, RMSE from squared errors, bias, and the same per count bin."""
    if len(bins) != len(metrics["abs_error"]):
        raise ValueError("one bin label per image")

    def block(chosen: list[int]) -> dict:
        index = torch.tensor(chosen, dtype=torch.long)
        if not len(index):
            return {"images": 0}
        mse = metrics["sq_error"][index].mean().item()
        return {
            "mae": round(metrics["abs_error"][index].mean().item(), 4),
            "rmse": round(math.sqrt(mse), 4),
            "bias": round(metrics["signed_error"][index].mean().item(), 4),
            "mean_true_count": round(metrics["true_count"][index].mean().item(), 4),
            "images": len(index),
        }

    rows = {name: [i for i, b in enumerate(bins) if b == name] for name in BINS}
    return block(list(range(len(bins)))) | {"by_bin": {name: block(r) for name, r in rows.items()}}


def bins_for(true_counts) -> list[str]:
    return [count_bin(int(c)) for c in true_counts]
