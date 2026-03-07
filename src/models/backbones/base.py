from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class BackboneOutput:
    feature_map: torch.Tensor
    mask: torch.Tensor


class BackboneBase(nn.Module):
    output_dim: int

    def forward(self, images: torch.Tensor) -> BackboneOutput:  # pragma: no cover - interface
        raise NotImplementedError
