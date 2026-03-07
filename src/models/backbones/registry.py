from __future__ import annotations

from src.models.backbones.base import BackboneBase
from src.models.backbones.clip import CLIPPatchBackbone, OpenCLIPBackbone
from src.models.backbones.dinov2 import DINOv2PatchBackbone, HubDINOv2Backbone
from src.models.backbones.resnet import ResNetFeatureBackbone, TorchvisionResNetBackbone


def build_backbone(name: str, pretrained: bool = False, enc_output_layer: int = -1) -> BackboneBase:
    if name in {"dinov2", "dinov2_q"}:
        if pretrained:
            return HubDINOv2Backbone(use_q_features=name == "dinov2_q", enc_output_layer=enc_output_layer)
        return DINOv2PatchBackbone()
    if name in {"clip", "clip_cls"}:
        if pretrained:
            return OpenCLIPBackbone()
        return CLIPPatchBackbone(output_dim=768)
    if name in {"resnet18", "resnet50"}:
        if pretrained:
            return TorchvisionResNetBackbone(name=name)
        return ResNetFeatureBackbone(output_dim=768)
    raise ValueError(f"Unsupported backbone '{name}'")
