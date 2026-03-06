from pathlib import Path

from src.config import load_config


def test_load_config_builds_run_directory(tmp_path: Path):
    config = load_config(
        "configs/tben/base.yaml",
        overrides={"output_root": str(tmp_path), "subject": 1},
    )
    assert config.subject == 1
    assert config.run_dir.parent == tmp_path
