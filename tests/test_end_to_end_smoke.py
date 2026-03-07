import json
import subprocess
import sys


def test_train_evaluate_predict_smoke(tmp_path):
    output_root = tmp_path / "runs"
    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.train",
            "--config",
            "configs/tben/base.yaml",
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
    assert train.returncode == 0
    checkpoint = output_root / "subj01_tben" / "last.pth"
    assert checkpoint.exists()

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.evaluate",
            "--config",
            "configs/tben/base.yaml",
            "--subject",
            "1",
            "--checkpoint",
            str(checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert evaluate.returncode == 0

    prediction_path = tmp_path / "prediction.json"
    predict = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.predict",
            "--config",
            "configs/tben/base.yaml",
            "--subject",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(prediction_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert predict.returncode == 0
    payload = json.loads(prediction_path.read_text())
    assert "lh_shape" in payload
