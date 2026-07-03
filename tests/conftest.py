# set the environment variables in the fixture BEFORE importing pytest
import os

os.environ["WANDB_MODE"] = "offline"
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests.",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all the tests to get the total coverage.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        # --integration given in CLI, so don't skip integration tests and skip the rest
        skip_non_integration = pytest.mark.skip(reason="--integration option is set")
        for item in items:
            if "integration" not in item.keywords:
                item.add_marker(skip_non_integration)
    elif config.getoption("--all"):
        # --all given in CLI, so don't skip any tests
        return
    else:
        # --integration given in CLI, so skip integration tests and don't skip the rest
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
