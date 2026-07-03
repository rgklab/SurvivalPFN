"""
Callback context for training callbacks.

This module provides a dataclass that encapsulates all the information passed
to callbacks during training, making the callback interface cleaner and easier to extend.
"""

import dataclasses

from torch.nn import Module
from torch.optim import Optimizer


@dataclasses.dataclass
class CallbackContext:
    """
    Context information passed to callbacks during training.

    This dataclass bundles all the parameters that callbacks need, providing
    a cleaner interface and making it easier to add new parameters without
    breaking existing callbacks.

    Attributes:
        callback_idx: Index of the callback in the callback list
        model: The model being trained
        optimizer: The optimizer used for training
        epoch: Current epoch number
        train_loss: Training loss for the current epoch
        device: Device on which the model is trained (e.g., 'cuda:0', 'cpu')
        wandb_enabled: Whether Weights & Biases logging is enabled
    """

    callback_idx: int
    model: Module
    optimizer: Optimizer
    epoch: int
    train_loss: float
    device: str
    wandb_enabled: bool
    callback_metrics: dict[str, float] = dataclasses.field(default_factory=dict)

    def is_main_process(self) -> bool:
        """Return True for the single training process."""
        return True

    def should_log(self) -> bool:
        """
        Check if this process should log to external services (e.g., wandb).

        Returns:
            True if wandb logging is enabled.
        """
        return self.wandb_enabled
