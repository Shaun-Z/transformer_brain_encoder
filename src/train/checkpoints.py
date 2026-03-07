from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = float("-inf")

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pth"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pth"

    def save_last(self, state: dict[str, Any]) -> None:
        torch.save(state, self.last_path)

    def save_best(self, state: dict[str, Any], score: float) -> None:
        if score >= self.best_score:
            self.best_score = score
            torch.save(state, self.best_path)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    def save_config(self, config: dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
