from pathlib import Path

from src.train.checkpoints import CheckpointManager


def test_checkpoint_manager_writes_last_and_best(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    manager.save_last({"epoch": 0})
    manager.save_best({"epoch": 0}, score=0.1)
    assert (tmp_path / "last.pth").exists()
    assert (tmp_path / "best.pth").exists()
