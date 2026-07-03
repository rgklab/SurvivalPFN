import os
from datetime import datetime

import torch
import wandb

from .base import Callback


class Checkpoint(Callback):
    """Store the top k models based on a monitored score."""

    SUPPORTED_MONITORS = {
        "train_loss": False,
        "avg_harrell_concordance": True,
        "avg_integrated_brier": False,
    }

    def __init__(
        self,
        output_path: str,
        callback_name: str | None,
        frequency: int = 1,
        checkpoint_dir_name: str | None = None,
        top_k: int = 2,
        monitor: str | list[str] = "train_loss",
    ):
        super().__init__(callback_name=callback_name)
        self.frequency = frequency
        self.output_path = output_path
        raw_monitors = [monitor] if isinstance(monitor, str) else list(monitor)
        self.monitors = list(dict.fromkeys(raw_monitors))
        if not self.monitors:
            raise ValueError("Checkpoint monitor list cannot be empty.")
        invalid_monitors = [
            metric for metric in self.monitors if metric not in self.SUPPORTED_MONITORS
        ]
        if invalid_monitors:
            raise ValueError(f"Unsupported checkpoint monitor(s): {invalid_monitors}")
        self.monitor = self.monitors[0] if len(self.monitors) == 1 else self.monitors

        if wandb.run is not None:
            self.default_name = f"wandb-{wandb.run.id}"
        else:
            self.default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.checkpoint_dir_name = checkpoint_dir_name or self.default_name
        os.makedirs(
            os.path.join(self.output_path, self.checkpoint_dir_name), exist_ok=True
        )
        self.top_k = top_k
        self.top_scores = {
            metric: [float("-inf") if self.SUPPORTED_MONITORS[metric] else float("inf")]
            * top_k
            for metric in self.monitors
        }
        self.top_epochs = {metric: [-1] * top_k for metric in self.monitors}

    @property
    def requires_eval(self) -> bool:
        return any(metric != "train_loss" for metric in self.monitors)

    def _get_monitor_value(self, metric: str, context) -> float | None:
        if metric == "train_loss":
            return context.train_loss
        return context.callback_metrics.get(metric)

    def _is_better(self, metric: str, score: float) -> bool:
        if self.SUPPORTED_MONITORS[metric]:
            return score > self.top_scores[metric][-1]
        return score < self.top_scores[metric][-1]

    def _checkpoint_path(self, metric: str, epoch: int) -> str:
        return os.path.join(
            self.output_path, self.checkpoint_dir_name, f"{metric}_epoch_{epoch:04d}.pt"
        )

    @torch.no_grad()
    def __call__(self, context):
        """
        Save model checkpoints based on a monitored score.

        Args:
            context: CallbackContext with training state
        """
        if (context.epoch + 1) % self.frequency == 0:
            monitor_values = {
                metric: self._get_monitor_value(metric, context)
                for metric in self.monitors
            }
            training_state = {
                "model_state_dict": getattr(
                    context.model, "_orig_mod", context.model
                ).state_dict(),
                "model_config": context.model.model_config,
                "optimizer_state_dict": context.optimizer.state_dict(),
                "epoch": context.epoch,
                "train_loss": context.train_loss,
                "monitor": self.monitor,
                "monitors": self.monitors,
                "monitor_values": monitor_values,
                "run_name": self.default_name,
            }

            for metric, monitor_value in monitor_values.items():
                if (
                    self.top_k <= 0
                    or monitor_value is None
                    or not self._is_better(metric, monitor_value)
                ):
                    continue

                removal_idx = self.top_epochs[metric][-1]
                if removal_idx != -1 and context.is_main_process():
                    os.remove(self._checkpoint_path(metric, removal_idx))

                self.top_epochs[metric][-1] = context.epoch
                self.top_scores[metric][-1] = monitor_value
                everything = list(zip(self.top_scores[metric], self.top_epochs[metric]))
                everything_sorted = sorted(
                    everything,
                    key=lambda x: x[0],
                    reverse=self.SUPPORTED_MONITORS[metric],
                )
                for i, (score, e) in enumerate(everything_sorted):
                    self.top_scores[metric][i] = score
                    self.top_epochs[metric][i] = e

                if context.is_main_process():
                    torch.save(
                        training_state, self._checkpoint_path(metric, context.epoch)
                    )
            if context.is_main_process():
                torch.save(
                    training_state,
                    os.path.join(
                        self.output_path, f"{self.checkpoint_dir_name}", "latest.pt"
                    ),
                )
