from pathlib import Path

import torch

from survivalpfn.callbacks import CallbackContext, Checkpoint
from survivalpfn.callbacks.eval import EvalSurvival
from train import _order_callbacks


def _make_context(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    callback_metrics: dict[str, float] | None = None,
) -> CallbackContext:
    return CallbackContext(
        callback_idx=0,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        train_loss=train_loss,
        device="cpu",
        wandb_enabled=False,
        callback_metrics=callback_metrics or {},
    )


def test_checkpoint_uses_train_loss_by_default(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    model.model_config = {"name": "dummy"}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = Checkpoint(
        output_path=str(tmp_path),
        callback_name="checkpoint",
        checkpoint_dir_name="run",
        top_k=1,
    )

    checkpoint(_make_context(model, optimizer, epoch=0, train_loss=1.0))
    checkpoint(_make_context(model, optimizer, epoch=1, train_loss=2.0))

    assert (tmp_path / "run" / "train_loss_epoch_0000.pt").exists()
    assert not (tmp_path / "run" / "train_loss_epoch_0001.pt").exists()


def test_checkpoint_can_monitor_validation_c_index(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    model.model_config = {"name": "dummy"}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = Checkpoint(
        output_path=str(tmp_path),
        callback_name="checkpoint",
        checkpoint_dir_name="run",
        top_k=1,
        monitor="avg_harrell_concordance",
    )

    checkpoint(_make_context(model, optimizer, epoch=0, train_loss=1.0))
    latest = torch.load(
        tmp_path / "run" / "latest.pt", weights_only=False, map_location="cpu"
    )
    assert latest["monitor"] == "avg_harrell_concordance"
    assert latest["monitor_values"]["avg_harrell_concordance"] is None
    assert not (tmp_path / "run" / "avg_harrell_concordance_epoch_0000.pt").exists()

    checkpoint(
        _make_context(
            model,
            optimizer,
            epoch=1,
            train_loss=2.0,
            callback_metrics={"avg_harrell_concordance": 0.6},
        )
    )
    checkpoint(
        _make_context(
            model,
            optimizer,
            epoch=2,
            train_loss=0.5,
            callback_metrics={"avg_harrell_concordance": 0.4},
        )
    )
    checkpoint(
        _make_context(
            model,
            optimizer,
            epoch=3,
            train_loss=3.0,
            callback_metrics={"avg_harrell_concordance": 0.8},
        )
    )

    assert not (tmp_path / "run" / "avg_harrell_concordance_epoch_0001.pt").exists()
    assert not (tmp_path / "run" / "avg_harrell_concordance_epoch_0002.pt").exists()
    assert (tmp_path / "run" / "avg_harrell_concordance_epoch_0003.pt").exists()

    best = torch.load(
        tmp_path / "run" / "avg_harrell_concordance_epoch_0003.pt",
        weights_only=False,
        map_location="cpu",
    )
    assert best["monitor"] == "avg_harrell_concordance"
    assert best["monitor_values"]["avg_harrell_concordance"] == 0.8


def test_checkpoint_can_track_multiple_metrics(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    model.model_config = {"name": "dummy"}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = Checkpoint(
        output_path=str(tmp_path),
        callback_name="checkpoint",
        checkpoint_dir_name="run",
        top_k=1,
        monitor=["train_loss", "avg_harrell_concordance", "avg_integrated_brier"],
    )

    checkpoint(_make_context(model, optimizer, epoch=0, train_loss=1.0))
    checkpoint(
        _make_context(
            model,
            optimizer,
            epoch=1,
            train_loss=2.0,
            callback_metrics={
                "avg_harrell_concordance": 0.6,
                "avg_integrated_brier": 0.4,
            },
        )
    )
    checkpoint(
        _make_context(
            model,
            optimizer,
            epoch=2,
            train_loss=0.5,
            callback_metrics={
                "avg_harrell_concordance": 0.4,
                "avg_integrated_brier": 0.5,
            },
        )
    )
    checkpoint(
        _make_context(
            model,
            optimizer,
            epoch=3,
            train_loss=0.7,
            callback_metrics={
                "avg_harrell_concordance": 0.8,
                "avg_integrated_brier": 0.3,
            },
        )
    )

    assert not (tmp_path / "run" / "train_loss_epoch_0000.pt").exists()
    assert (tmp_path / "run" / "train_loss_epoch_0002.pt").exists()
    assert not (tmp_path / "run" / "avg_harrell_concordance_epoch_0001.pt").exists()
    assert (tmp_path / "run" / "avg_harrell_concordance_epoch_0003.pt").exists()
    assert not (tmp_path / "run" / "avg_integrated_brier_epoch_0001.pt").exists()
    assert (tmp_path / "run" / "avg_integrated_brier_epoch_0003.pt").exists()

    latest = torch.load(
        tmp_path / "run" / "latest.pt", weights_only=False, map_location="cpu"
    )
    assert latest["monitors"] == [
        "train_loss",
        "avg_harrell_concordance",
        "avg_integrated_brier",
    ]
    assert latest["monitor_values"] == {
        "train_loss": 0.7,
        "avg_harrell_concordance": 0.8,
        "avg_integrated_brier": 0.3,
    }


def test_validation_monitor_moves_eval_before_checkpoint(tmp_path: Path):
    checkpoint = Checkpoint(
        output_path=str(tmp_path),
        callback_name="checkpoint",
        checkpoint_dir_name="run",
        top_k=1,
        monitor=["train_loss", "avg_integrated_brier"],
    )
    eval_callback = object.__new__(EvalSurvival)

    ordered_callbacks = _order_callbacks([checkpoint, eval_callback])

    assert isinstance(ordered_callbacks[0], EvalSurvival)
    assert ordered_callbacks[1] is checkpoint
