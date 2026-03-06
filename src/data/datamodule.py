from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.dataset import AlgonautsDataset, collate_samples
from src.data.manifest import SubjectManifest, build_subject_manifest


@dataclass(slots=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    manifest: SubjectManifest

    def __contains__(self, key: str) -> bool:
        return key in {"train", "val", "manifest"}

    def __getitem__(self, key: str):
        return getattr(self, key)


def build_train_val_indices(total_items: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if total_items < 2:
        raise ValueError("Need at least 2 training samples to create train/val splits.")

    indices = np.arange(total_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(total_items * val_ratio)))
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    if train_indices.size == 0:
        raise ValueError("Validation split consumed the full dataset.")
    return train_indices, val_indices


def build_dataloaders(config: ExperimentConfig) -> DataLoaders:
    manifest = build_subject_manifest(config.dataset_root, subject=config.subject)
    total_items = len(list(manifest.training_images_dir.glob("*.png")))
    train_indices, val_indices = build_train_val_indices(total_items, config.val_ratio, config.seed)

    train_dataset = AlgonautsDataset(manifest, train_indices)
    val_dataset = AlgonautsDataset(manifest, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )
    return DataLoaders(train=train_loader, val=val_loader, manifest=manifest)
