from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.models.backbones.base import BackboneBase, BackboneOutput
from src.models.backbones.dinov2 import DINOv2PatchBackbone


class ResNetFeatureBackbone(DINOv2PatchBackbone):
    pass


class TorchvisionResNetBackbone(BackboneBase):
    def __init__(self, name: str = "resnet50") -> None:
        super().__init__()
        backbone = getattr(models, name)(weights=None)
        layers = list(backbone.children())[:-2]
        self.body = nn.Sequential(*layers)
        self.output_dim = 2048 if name == "resnet50" else 512
        for parameter in self.body.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> BackboneOutput:
        feature_map = self.body(images)
        mask = torch.zeros(
            (feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]),
            dtype=torch.bool,
            device=feature_map.device,
        )
        return BackboneOutput(feature_map=feature_map, mask=mask)
