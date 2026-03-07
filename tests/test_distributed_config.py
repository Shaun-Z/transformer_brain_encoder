from src.config import load_config


def test_config_can_enable_distributed():
    config = load_config("configs/tben/base.yaml", overrides={"distributed": True})
    assert config.distributed is True
