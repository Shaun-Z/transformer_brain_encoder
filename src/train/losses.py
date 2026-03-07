from __future__ import annotations

import torch
from torch import nn

from src.data.rois import RoiBundle


def aggregate_predictions(outputs: dict[str, torch.Tensor], roi_bundle: RoiBundle) -> tuple[torch.Tensor, torch.Tensor]:
    lh_pred = outputs["lh_f_pred"]
    rh_pred = outputs["rh_f_pred"]

    if lh_pred.ndim == 3:
        lh_pred = torch.einsum("bvq,qv->bv", lh_pred, roi_bundle.lh_challenge_rois.float().to(lh_pred.device))
    if rh_pred.ndim == 3:
        rh_pred = torch.einsum("bvq,qv->bv", rh_pred, roi_bundle.rh_challenge_rois.float().to(rh_pred.device))
    return lh_pred, rh_pred


class BrainEncodingLoss(nn.Module):
    def __init__(self, roi_bundle: RoiBundle | None = None) -> None:
        super().__init__()
        self.roi_bundle = roi_bundle
        self.mse = nn.MSELoss()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.roi_bundle is None:
            lh_pred = outputs["lh_f_pred"]
            rh_pred = outputs["rh_f_pred"]
        else:
            lh_pred, rh_pred = aggregate_predictions(outputs, self.roi_bundle)

        return self.mse(lh_pred, batch["lh_fmri"]) + self.mse(rh_pred, batch["rh_fmri"])
