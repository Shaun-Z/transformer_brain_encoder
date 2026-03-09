from __future__ import annotations

from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.data.manifest import build_subject_manifest
from src.data.rois import load_roi_bundle
from src.models.model import build_model
from src.train.losses import aggregate_predictions


class brain_encoder_wrapper:
    def __init__(
        self,
        checkpoint_path: str,
        dataset_root: str = "/data/algonauts_2023_challenge_data",
        subject: int = 1,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        config_data = checkpoint["config"]
        self.config = ExperimentConfig(
            config_name="loaded",
            output_root=Path(config_data["output_root"]),
            run_name=config_data["run_name"],
            subject=int(config_data["subject"]),
            run_dir=Path(config_data["run_dir"]),
            dataset_root=Path(dataset_root),
            distributed=bool(config_data["distributed"]),
            batch_size=int(config_data["batch_size"]),
            num_workers=int(config_data["num_workers"]),
            val_ratio=float(config_data["val_ratio"]),
            seed=int(config_data["seed"]),
            pretrained_backbone=bool(config_data["pretrained_backbone"]),
            hidden_dim=int(config_data["hidden_dim"]),
            decoder_layers=int(config_data["decoder_layers"]),
            encoder_layers=int(config_data["encoder_layers"]),
            nheads=int(config_data["nheads"]),
            dim_feedforward=int(config_data["dim_feedforward"]),
            dropout=float(config_data["dropout"]),
            backbone_name=config_data["backbone_name"],
            readout_name=config_data["readout_name"],
        )
        manifest = build_subject_manifest(dataset_root, subject=subject)
        self.roi_bundle = load_roi_bundle(manifest, "rois_all")
        self.model = build_model(
            backbone_name=self.config.backbone_name,
            readout_name=self.config.readout_name,
            hidden_dim=self.config.hidden_dim,
            lh_output_dim=self.roi_bundle.lh_challenge_rois.shape[1],
            rh_output_dim=self.roi_bundle.rh_challenge_rois.shape[1],
            lh_query_count=self.roi_bundle.lh_challenge_rois.shape[0],
            rh_query_count=self.roi_bundle.rh_challenge_rois.shape[0],
            pretrained_backbone=self.config.pretrained_backbone,
            enc_output_layer=self.config.enc_output_layer,
        )
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        outputs = self.model(images)
        lh_pred, rh_pred = aggregate_predictions(outputs, self.roi_bundle)
        return lh_pred, rh_pred
