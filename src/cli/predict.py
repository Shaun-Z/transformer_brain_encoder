from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import load_config
from src.train.trainer import build_training_components


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prediction on the validation loader")
    parser.add_argument("--config", default="configs/tben/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", type=int)
    args = parser.parse_args()

    overrides = {}
    if args.subject is not None:
        overrides["subject"] = args.subject
    config = load_config(args.config, overrides=overrides)
    dataloaders, roi_bundle, model = build_training_components(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    batch = next(iter(dataloaders.val))
    with torch.no_grad():
        outputs = model(batch["image"])

    payload = {
        "lh_shape": list(outputs["lh_f_pred"].shape),
        "rh_shape": list(outputs["rh_f_pred"].shape),
        "roi_queries": {
            "lh": int(roi_bundle.lh_challenge_rois.shape[0]),
            "rh": int(roi_bundle.rh_challenge_rois.shape[0]),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
