from __future__ import annotations

from torch import nn

from src.models.readouts.linear import LinearReadout
from src.models.readouts.spatial_feature import SpatialFeatureReadout
from src.models.readouts.tben import TBenReadout


def build_readout(
    name: str,
    feature_dim: int,
    hidden_dim: int,
    lh_output_dim: int,
    rh_output_dim: int,
    lh_query_count: int,
    rh_query_count: int,
) -> nn.Module:
    if name == "tben":
        return TBenReadout(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            lh_output_dim=lh_output_dim,
            rh_output_dim=rh_output_dim,
            lh_query_count=lh_query_count,
            rh_query_count=rh_query_count,
        )
    if name == "linear":
        return LinearReadout(feature_dim=feature_dim, lh_output_dim=lh_output_dim, rh_output_dim=rh_output_dim)
    if name == "spatial_feature":
        return SpatialFeatureReadout(
            feature_dim=feature_dim,
            lh_output_dim=lh_output_dim,
            rh_output_dim=rh_output_dim,
            lh_query_count=lh_query_count,
            rh_query_count=rh_query_count,
        )
    raise ValueError(f"Unsupported readout '{name}'")
