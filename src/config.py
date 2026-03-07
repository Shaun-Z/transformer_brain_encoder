from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    config_name: str
    output_root: Path
    run_name: str
    subject: int
    run_dir: Path
    dataset_root: Path
    distributed: bool
    batch_size: int
    num_workers: int
    val_ratio: float
    seed: int
    pretrained_backbone: bool
    hidden_dim: int
    decoder_layers: int
    encoder_layers: int
    enc_output_layer: int
    nheads: int
    dim_feedforward: int
    dropout: float
    backbone_name: str
    readout_name: str


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten_override_key(data: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    if "." not in key:
        data[key] = value
        return data

    head, tail = key.split(".", 1)
    nested = data.setdefault(head, {})
    if not isinstance(nested, dict):
        nested = {}
        data[head] = nested
    _flatten_override_key(nested, tail, value)
    return data


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(config_path)
    config_data = yaml.safe_load(path.read_text()) or {}
    overrides = overrides or {}

    normalized_overrides: dict[str, Any] = {}
    for key, value in overrides.items():
        _flatten_override_key(normalized_overrides, key, value)

    config_data = _deep_update(config_data, normalized_overrides)

    output_root = Path(config_data["output_root"]).expanduser().resolve()
    subject = int(config_data["subject"])
    run_name = config_data.get("run_name", f"subj{subject:02d}_{config_data['backbone_name']}_{config_data['readout_name']}")
    run_dir = output_root / run_name

    return ExperimentConfig(
        config_name=path.stem,
        output_root=output_root,
        run_name=run_name,
        subject=subject,
        run_dir=run_dir,
        dataset_root=Path(config_data["dataset_root"]).expanduser().resolve(),
        distributed=bool(config_data.get("distributed", False)),
        batch_size=int(config_data.get("batch_size", 16)),
        num_workers=int(config_data.get("num_workers", 4)),
        val_ratio=float(config_data.get("val_ratio", 0.1)),
        seed=int(config_data.get("seed", 0)),
        pretrained_backbone=bool(config_data.get("pretrained_backbone", False)),
        hidden_dim=int(config_data.get("hidden_dim", 256)),
        decoder_layers=int(config_data.get("decoder_layers", 1)),
        encoder_layers=int(config_data.get("encoder_layers", 0)),
        enc_output_layer=int(config_data.get("enc_output_layer", -1)),
        nheads=int(config_data.get("nheads", 8)),
        dim_feedforward=int(config_data.get("dim_feedforward", 1024)),
        dropout=float(config_data.get("dropout", 0.1)),
        backbone_name=config_data["backbone_name"],
        readout_name=config_data["readout_name"],
    )
