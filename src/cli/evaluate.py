from __future__ import annotations

import argparse

import torch

from src.config import load_config
from src.train.trainer import build_training_components, evaluate_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained run")
    parser.add_argument("--config", default="configs/tben/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", type=int)
    args = parser.parse_args()

    overrides = {}
    if args.subject is not None:
        overrides["subject"] = args.subject
    config = load_config(args.config, overrides=overrides)
    dataloaders, roi_bundle, model = build_training_components(config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    metrics = evaluate_model(model, dataloaders.val, roi_bundle, torch.device("cpu"))
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
