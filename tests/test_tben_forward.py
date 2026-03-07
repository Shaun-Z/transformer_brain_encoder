import torch

from src.models.model import build_model


def test_tben_forward_shapes():
    model = build_model(
        backbone_name="dinov2_q",
        readout_name="tben",
        hidden_dim=32,
        lh_output_dim=7,
        rh_output_dim=9,
        lh_query_count=3,
        rh_query_count=4,
    )
    images = torch.randn(2, 3, 224, 224)
    outputs = model(images)
    assert "lh_f_pred" in outputs
    assert "rh_f_pred" in outputs
