from src.models.readouts.registry import build_readout


def test_build_linear_readout():
    readout = build_readout(
        "linear",
        feature_dim=32,
        hidden_dim=32,
        lh_output_dim=7,
        rh_output_dim=9,
        lh_query_count=1,
        rh_query_count=1,
    )
    assert readout is not None
