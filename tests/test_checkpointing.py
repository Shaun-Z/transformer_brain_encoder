from pathlib import Path

from src.train.checkpoints import CheckpointManager, checkpoint_best_state, checkpoint_model_state


def test_checkpoint_manager_writes_last_and_best(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    manager.save_last({"epoch": 0})
    manager.save_best({"epoch": 0}, score=0.1)
    assert (tmp_path / "last.pth").exists()
    assert (tmp_path / "best.pth").exists()


def test_checkpoint_model_state_drops_backbone_weights():
    state_dict = {
        "backbone.proj.weight": 1,
        "backbone.proj.bias": 2,
        "readout.lh_head.weight": 3,
    }
    checkpoint_state = checkpoint_model_state(state_dict)
    assert "backbone.proj.weight" not in checkpoint_state
    assert "backbone.proj.bias" not in checkpoint_state
    assert checkpoint_state["readout.lh_head.weight"] == 3


def test_checkpoint_best_state_drops_optimizer():
    state = {"epoch": 0, "optimizer": {"lr": 1e-4}, "metrics": {"val_corr": 0.1}}
    best_state = checkpoint_best_state(state)
    assert "optimizer" not in best_state
    assert best_state["epoch"] == 0
