from pathlib import Path

from src.config import load_config


def test_load_config_builds_run_directory(tmp_path: Path):
    config = load_config(
        "configs/tben/base.yaml",
        overrides={"output_root": str(tmp_path), "subject": 1},
    )
    assert config.subject == 1
    assert config.run_dir.parent == tmp_path


def test_subject_override_changes_run_name_and_run_dir(tmp_path: Path):
    config = load_config(
        "configs/tben/dinov2_q_rois_all.yaml",
        overrides={"output_root": str(tmp_path), "subject": 6},
    )
    assert config.subject == 6
    assert config.run_name == "subj06_dinov2_q_rois_all"
    assert config.run_dir == tmp_path / "subj06_dinov2_q_rois_all"
