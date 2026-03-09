from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import load_config
from src.train.trainer import build_training_components, evaluate_model


def evaluate_subject(subject: int, config_path: str, results_root: Path, reports_root: Path) -> dict:
    config = load_config(config_path, overrides={"subject": subject, "output_root": str(results_root)})
    run_dir = results_root / config.run_name
    checkpoint_path = run_dir / "best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    dataloaders, roi_bundle, model = build_training_components(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    metrics = evaluate_model(model, dataloaders.val, roi_bundle, torch.device("cpu"))

    report = {
        "subject": subject,
        "checkpoint": str(checkpoint_path.resolve()),
        "metrics": metrics,
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"subj{subject:02d}_best_eval.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved best checkpoints and write reports")
    parser.add_argument("--config", default="configs/tben/dinov2_q_rois_all.yaml")
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--reports-root", default="./results/eval_reports")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 9)))
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    summary = []
    for subject in args.subjects:
        report = evaluate_subject(subject, args.config, results_root, reports_root)
        summary.append(report)
        print(json.dumps(report, sort_keys=True))

    summary_path = reports_root / "summary_best_eval.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
