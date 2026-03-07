from __future__ import annotations

import torch


def mean_vertex_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = predictions.float()
    targets = targets.float()
    predictions = predictions - predictions.mean(dim=0, keepdim=True)
    targets = targets - targets.mean(dim=0, keepdim=True)
    numerator = (predictions * targets).sum(dim=0)
    denominator = torch.sqrt((predictions.square().sum(dim=0) * targets.square().sum(dim=0)).clamp_min(1e-8))
    correlations = numerator / denominator
    correlations = torch.nan_to_num(correlations, nan=0.0)
    return float(correlations.mean().item())
