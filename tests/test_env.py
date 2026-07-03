# tests/test_env.py

import os


def test_environment_variables():
    assert os.getenv("WANDB_MODE") == "offline", "WANDB_MODE is not set to 'offline'."
