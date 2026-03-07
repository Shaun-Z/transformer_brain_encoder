from __future__ import annotations

import torch
from torch import nn


class SpatialFeatureReadout(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        lh_output_dim: int,
        rh_output_dim: int,
        lh_query_count: int,
        rh_query_count: int,
    ) -> None:
        super().__init__()
        self.lh_query_count = lh_query_count
        self.rh_query_count = rh_query_count
        self.total_queries = lh_query_count + rh_query_count
        self.query_embed = nn.Embedding(self.total_queries, feature_dim)
        self.lh_head = nn.Linear(feature_dim, lh_output_dim)
        self.rh_head = nn.Linear(feature_dim, rh_output_dim)

    def forward(self, feature_map: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del mask
        pooled = feature_map.flatten(2).mean(-1)
        queries = self.query_embed.weight.unsqueeze(0).expand(feature_map.shape[0], -1, -1)
        mixed = queries + pooled.unsqueeze(1)
        lh_tokens = mixed[:, : self.lh_query_count, :]
        rh_tokens = mixed[:, self.lh_query_count :, :]
        return {
            "lh_f_pred": self.lh_head(lh_tokens).transpose(1, 2),
            "rh_f_pred": self.rh_head(rh_tokens).transpose(1, 2),
            "output_tokens": mixed,
        }
