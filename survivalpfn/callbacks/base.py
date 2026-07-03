from abc import ABC, abstractmethod

from .context import CallbackContext


class Callback(ABC):
    """The base callback class that is used in training"""

    def __init__(self, callback_name: str | None):
        self.callback_name = callback_name or self.__class__.__name__

    @abstractmethod
    def __call__(self, context: CallbackContext):
        """
        The method that is called during training after each epoch.

        Args:
            context: CallbackContext object containing all training state information
                    including model, optimizer, epoch, loss, and device.
        """
        pass
