from src.models.backbones.registry import build_backbone


def test_build_backbone_dinov2_q():
    backbone = build_backbone("dinov2_q")
    assert backbone is not None
