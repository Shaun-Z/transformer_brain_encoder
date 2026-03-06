from src.config import load_config
from src.data.datamodule import build_dataloaders


def test_build_dataloaders_returns_train_and_val():
    config = load_config(
        "configs/tben/base.yaml",
        overrides={"subject": 1, "batch_size": 2, "num_workers": 0},
    )
    dataloaders = build_dataloaders(config)
    assert "train" in dataloaders
    assert "val" in dataloaders
