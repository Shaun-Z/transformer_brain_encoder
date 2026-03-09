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


def test_predict_works_with_pruned_checkpoint(tmp_path):
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
    assert predict.returncode == 0, predict.stderr

    payload = json.loads(prediction_path.read_text())
    assert "lh_shape" in payload
    assert "rh_shape" in payload


def test_export_prediction_shapes_reports(tmp_path):
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

    reports_root = tmp_path / "prediction_reports"
    export = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tools.export_prediction_shapes",
            "--config",
            "configs/tben/base.yaml",
            "--results-root",
            str(output_root),
            "--reports-root",
            str(reports_root),
            "--subjects",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert export.returncode == 0, export.stderr

    report_path = reports_root / "subj01_best_prediction_shape.json"
    summary_path = reports_root / "summary_best_prediction_shapes.json"
    assert report_path.exists()
    assert summary_path.exists()

    report = json.loads(report_path.read_text())
    assert report["subject"] == 1
    assert "lh_shape" in report
    assert "rh_shape" in report
