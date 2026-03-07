import subprocess
import sys


def test_inspect_data_cli_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.inspect_data",
            "--dataset-root",
            "/data/algonauts_2023_challenge_data",
            "--subject",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
