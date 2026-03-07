from __future__ import annotations

import torch
from torch import nn


class LinearReadout(nn.Module):
    def __init__(self, feature_dim: int, lh_output_dim: int, rh_output_dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lh_head = nn.Linear(feature_dim, lh_output_dim)
        self.rh_head = nn.Linear(feature_dim, rh_output_dim)

    def forward(self, feature_map: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del mask
        pooled = self.pool(feature_map).flatten(1)
        return {
            "lh_f_pred": self.lh_head(pooled),
            "rh_f_pred": self.rh_head(pooled),
            "output_tokens": pooled,
        }
