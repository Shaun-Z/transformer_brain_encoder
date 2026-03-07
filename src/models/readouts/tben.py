from __future__ import annotations

import torch
from torch import nn


def _build_2d_sine_position_encoding(height: int, width: int, channels: int, device: torch.device) -> torch.Tensor:
    if channels % 4 != 0:
        raise ValueError("channels must be divisible by 4 for 2D sine position encoding")

    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    omega = torch.arange(channels // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, channels // 4 - 1)))

    y = y.reshape(-1, 1).float() * omega.reshape(1, -1)
    x = x.reshape(-1, 1).float() * omega.reshape(1, -1)
    pos = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pos


class TBenReadout(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        lh_output_dim: int,
        rh_output_dim: int,
        lh_query_count: int,
        rh_query_count: int,
        decoder_layers: int = 1,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lh_query_count = lh_query_count
        self.rh_query_count = rh_query_count
        self.total_queries = lh_query_count + rh_query_count

        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.query_embed = nn.Embedding(self.total_queries, hidden_dim)
        self.lh_head = nn.Linear(hidden_dim, lh_output_dim)
        self.rh_head = nn.Linear(hidden_dim, rh_output_dim)

    def forward(self, feature_map: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del mask
        projected = self.input_proj(feature_map)
        batch_size, hidden_dim, height, width = projected.shape
        memory = projected.flatten(2).permute(2, 0, 1)
        pos = _build_2d_sine_position_encoding(height, width, hidden_dim, projected.device)
        memory = memory + pos.unsqueeze(1)

        queries = self.query_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)
        targets = torch.zeros_like(queries)
        decoded = self.decoder(tgt=targets + queries, memory=memory).permute(1, 0, 2)

        lh_tokens = decoded[:, : self.lh_query_count, :]
        rh_tokens = decoded[:, self.lh_query_count :, :]
        lh_f_pred = self.lh_head(lh_tokens).transpose(1, 2)
        rh_f_pred = self.rh_head(rh_tokens).transpose(1, 2)
        return {
            "lh_f_pred": lh_f_pred,
            "rh_f_pred": rh_f_pred,
            "output_tokens": decoded,
        }
