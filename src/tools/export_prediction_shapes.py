from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import load_config
from src.train.trainer import build_training_components


def export_subject_prediction_shape(
    subject: int,
    config_path: str,
    results_root: Path,
    reports_root: Path,
) -> dict:
    config = load_config(config_path, overrides={"subject": subject, "output_root": str(results_root)})
    checkpoint_path = results_root / config.run_name / "best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    dataloaders, roi_bundle, model = build_training_components(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    batch = next(iter(dataloaders.val))
    with torch.no_grad():
        outputs = model(batch["image"])

    report = {
        "subject": subject,
        "checkpoint": str(checkpoint_path.resolve()),
        "lh_shape": list(outputs["lh_f_pred"].shape),
        "rh_shape": list(outputs["rh_f_pred"].shape),
        "roi_queries": {
            "lh": int(roi_bundle.lh_challenge_rois.shape[0]),
            "rh": int(roi_bundle.rh_challenge_rois.shape[0]),
        },
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"subj{subject:02d}_best_prediction_shape.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export prediction shapes for saved best checkpoints")
    parser.add_argument("--config", default="configs/tben/dinov2_q_rois_all.yaml")
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--reports-root", default="./results/prediction_reports")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 9)))
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    summary = []
    for subject in args.subjects:
        report = export_subject_prediction_shape(subject, args.config, results_root, reports_root)
        summary.append(report)
        print(json.dumps(report, sort_keys=True))

    summary_path = reports_root / "summary_best_prediction_shapes.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
