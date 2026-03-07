from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config import ExperimentConfig
from src.data.datamodule import build_dataloaders
from src.data.rois import RoiBundle, load_roi_bundle
from src.models.model import build_model
from src.train.checkpoints import CheckpointManager
from src.train.distributed import cleanup_distributed, is_main_process, setup_distributed
from src.train.losses import BrainEncodingLoss, aggregate_predictions
from src.train.metrics import mean_vertex_correlation


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_training_components(config: ExperimentConfig):
    dataloaders = build_dataloaders(config)
    roi_bundle = load_roi_bundle(dataloaders.manifest, "rois_all")

    model = build_model(
        backbone_name=config.backbone_name,
        readout_name=config.readout_name,
        hidden_dim=config.hidden_dim,
        lh_output_dim=roi_bundle.lh_challenge_rois.shape[1],
        rh_output_dim=roi_bundle.rh_challenge_rois.shape[1],
        lh_query_count=roi_bundle.lh_challenge_rois.shape[0],
        rh_query_count=roi_bundle.rh_challenge_rois.shape[0],
        pretrained_backbone=config.pretrained_backbone,
        enc_output_layer=config.enc_output_layer,
    )
    return dataloaders, roi_bundle, model


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    roi_bundle: RoiBundle,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    lh_predictions = []
    rh_predictions = []
    lh_targets = []
    rh_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            lh_pred, rh_pred = aggregate_predictions(outputs, roi_bundle)
            lh_predictions.append(lh_pred.cpu())
            rh_predictions.append(rh_pred.cpu())
            lh_targets.append(batch["lh_fmri"].cpu())
            rh_targets.append(batch["rh_fmri"].cpu())

    lh_predictions = torch.cat(lh_predictions, dim=0)
    rh_predictions = torch.cat(rh_predictions, dim=0)
    lh_targets = torch.cat(lh_targets, dim=0)
    rh_targets = torch.cat(rh_targets, dim=0)
    return {
        "lh_val_corr": mean_vertex_correlation(lh_predictions, lh_targets),
        "rh_val_corr": mean_vertex_correlation(rh_predictions, rh_targets),
        "val_corr": (mean_vertex_correlation(lh_predictions, lh_targets) + mean_vertex_correlation(rh_predictions, rh_targets)) / 2,
    }


def train(
    config: ExperimentConfig,
    epochs: int = 1,
    max_train_steps: int | None = None,
    device: str | torch.device | None = None,
) -> Path:
    distributed, rank, world_size, local_rank = setup_distributed(config.distributed)
    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataloaders = build_dataloaders(config, distributed=distributed, rank=rank, world_size=world_size)
    roi_bundle = load_roi_bundle(dataloaders.manifest, "rois_all")
    model = build_model(
        backbone_name=config.backbone_name,
        readout_name=config.readout_name,
        hidden_dim=config.hidden_dim,
        lh_output_dim=roi_bundle.lh_challenge_rois.shape[1],
        rh_output_dim=roi_bundle.rh_challenge_rois.shape[1],
        lh_query_count=roi_bundle.lh_challenge_rois.shape[0],
        rh_query_count=roi_bundle.rh_challenge_rois.shape[0],
        pretrained_backbone=config.pretrained_backbone,
        enc_output_layer=config.enc_output_layer,
    )
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = BrainEncodingLoss(roi_bundle=roi_bundle)
    checkpoint_manager = CheckpointManager(config.run_dir)
    if is_main_process(rank):
        checkpoint_manager.save_config({key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()})

    global_step = 0
    try:
        for epoch in range(epochs):
            if distributed and hasattr(dataloaders.train.sampler, "set_epoch"):
                dataloaders.train.sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            for batch in dataloaders.train:
                batch = move_batch_to_device(batch, device)
                outputs = model(batch["image"])
                loss = criterion(outputs, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                global_step += 1
                if max_train_steps is not None and global_step >= max_train_steps:
                    break

            metrics = {"train_loss": epoch_loss / max(1, global_step)}
            if is_main_process(rank):
                unwrapped_model = model.module if isinstance(model, DDP) else model
                val_metrics = evaluate_model(unwrapped_model, dataloaders.val, roi_bundle, device)
                metrics.update(val_metrics)
                state = {
                    "epoch": epoch,
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                    "metrics": metrics,
                }
                checkpoint_manager.save_last(state)
                checkpoint_manager.save_best(state, score=metrics["val_corr"])
                checkpoint_manager.save_metrics(metrics)
            if max_train_steps is not None and global_step >= max_train_steps:
                break
    finally:
        cleanup_distributed()

    return checkpoint_manager.last_path
