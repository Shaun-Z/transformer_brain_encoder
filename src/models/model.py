from __future__ import annotations

import torch
from torch import nn

from src.models.backbones.registry import build_backbone
from src.models.readouts.registry import build_readout


class BrainEncodingModel(nn.Module):
    def __init__(self, backbone: nn.Module, readout: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout = readout

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_output = self.backbone(images)
        return self.readout(backbone_output.feature_map, backbone_output.mask)


def build_model(
    backbone_name: str,
    readout_name: str,
    hidden_dim: int,
    lh_output_dim: int,
    rh_output_dim: int,
    lh_query_count: int,
    rh_query_count: int,
    pretrained_backbone: bool = False,
    enc_output_layer: int = -1,
) -> BrainEncodingModel:
    backbone = build_backbone(backbone_name, pretrained=pretrained_backbone, enc_output_layer=enc_output_layer)
    readout = build_readout(
        name=readout_name,
        feature_dim=backbone.output_dim,
        hidden_dim=hidden_dim,
        lh_output_dim=lh_output_dim,
        rh_output_dim=rh_output_dim,
        lh_query_count=lh_query_count,
        rh_query_count=rh_query_count,
    )
    return BrainEncodingModel(backbone=backbone, readout=readout)
