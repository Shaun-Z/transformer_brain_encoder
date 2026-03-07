import subprocess
import sys


def test_dinov2_q_paper_config_train_smoke(tmp_path):
    output_root = tmp_path / "paper-runs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.train",
            "--config",
            "configs/tben/dinov2_q_rois_all.yaml",
            "--subject",
            "1",
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--max-train-steps",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (output_root / "subj01_dinov2_q_rois_all" / "last.pth").exists()
