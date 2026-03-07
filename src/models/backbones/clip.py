from __future__ import annotations

import open_clip
import torch

from src.models.backbones.base import BackboneBase, BackboneOutput
from src.models.backbones.dinov2 import DINOv2PatchBackbone


class CLIPPatchBackbone(DINOv2PatchBackbone):
    pass


class OpenCLIPBackbone(BackboneBase):
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai") -> None:
        super().__init__()
        self.output_dim = 768
        self.backbone, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.backbone.visual.output_tokens = True
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> BackboneOutput:
        cls_token, patch_tokens = self.backbone.visual(images)
        del cls_token
        projected = patch_tokens @ self.backbone.visual.proj
        height = images.shape[-2] // 14
        width = images.shape[-1] // 14
        feature_map = projected.reshape(projected.shape[0], height, width, self.output_dim).permute(0, 3, 1, 2)
        mask = torch.zeros((feature_map.shape[0], height, width), dtype=torch.bool, device=feature_map.device)
        return BackboneOutput(feature_map=feature_map, mask=mask)
