from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.backbones.base import BackboneBase, BackboneOutput


class DINOv2PatchBackbone(BackboneBase):
    def __init__(self, output_dim: int = 768, patch_size: int = 14) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Conv2d(3, output_dim, kernel_size=patch_size, stride=patch_size)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> BackboneOutput:
        feature_map = self.proj(images)
        mask = torch.zeros(
            (feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]),
            dtype=torch.bool,
            device=feature_map.device,
        )
        return BackboneOutput(feature_map=feature_map, mask=mask)


class HubDINOv2Backbone(BackboneBase):
    def __init__(self, use_q_features: bool, enc_output_layer: int = -1, model_name: str = "dinov2_vitb14") -> None:
        super().__init__()
        self.output_dim = 768
        self.use_q_features = use_q_features
        self.enc_output_layer = enc_output_layer
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self._qkv_output: torch.Tensor | None = None
        if self.use_q_features:
            block_index = enc_output_layer if enc_output_layer >= 0 else len(self.backbone.blocks) + enc_output_layer
            self.backbone.blocks[block_index].attn.qkv.register_forward_hook(self._capture_qkv)

    def _capture_qkv(self, module, inputs, output) -> None:
        del module, inputs
        self._qkv_output = output

    def forward(self, images: torch.Tensor) -> BackboneOutput:
        patch_size = 14
        height = images.shape[-2] // patch_size
        width = images.shape[-1] // patch_size

        if self.use_q_features:
            _ = self.backbone.get_intermediate_layers(images, n=1)
            if self._qkv_output is None:
                raise RuntimeError("QKV hook did not capture any features.")
            qkv = self._qkv_output
            batch_size, sequence_length, three_dim = qkv.shape
            num_heads = self.backbone.blocks[0].attn.num_heads
            head_dim = three_dim // (3 * num_heads)
            qkv = qkv.reshape(batch_size, sequence_length, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            queries = qkv[0].transpose(1, 2).reshape(batch_size, sequence_length, self.output_dim)
            tokens = queries[:, 1:, :]
        else:
            tokens = self.backbone.get_intermediate_layers(images, n=1)[0][:, 1:, :]

        feature_map = tokens.reshape(tokens.shape[0], height, width, self.output_dim).permute(0, 3, 1, 2)
        mask = torch.zeros((feature_map.shape[0], height, width), dtype=torch.bool, device=feature_map.device)
        return BackboneOutput(feature_map=feature_map, mask=mask)
