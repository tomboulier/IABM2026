"""
Training Tracker Interface.

This module defines the abstract interface for tracking training progress
and metrics. Implementations can provide console output, MLflow logging,
TensorBoard integration, or other observability solutions.
"""
from abc import ABC, abstractmethod
from typing import Any


class TrainingTracker(ABC):
    """
    Abstract interface for tracking model training progress.

    This interface enables observability during training through a callback
    pattern. Implementations can log to console, MLflow, TensorBoard, or
    other backends.

    The lifecycle follows:
    1. on_training_start() - called once at the beginning
    2. For each epoch:
       a. on_epoch_start()
       b. For each batch: on_batch_end()
       c. on_epoch_end()
    3. on_training_end() - called once at the end
    """

    @abstractmethod
    def on_training_start(
        self,
        total_epochs: int,
        total_batches: int,
        dataset_name: str | None = None,
    ) -> None:
        """
        Called when training begins.

        Parameters
        ----------
        total_epochs : int
            Total number of epochs to train.
        total_batches : int
            Total number of batches per epoch.
        dataset_name : str | None, optional
            Name of the dataset being trained on.
        """
        pass

    @abstractmethod
    def on_epoch_start(self, epoch: int) -> None:
        """
        Called at the start of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        """
        pass

    @abstractmethod
    def on_batch_end(
        self,
        epoch: int,
        batch: int,
        loss: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Called after each batch completes.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        batch : int
            Current batch number (0-indexed).
        loss : float
            Loss value for this batch.
        metrics : dict[str, Any] | None, optional
            Additional metrics to track.
        """
        pass

    @abstractmethod
    def on_epoch_end(
        self,
        epoch: int,
        avg_loss: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        avg_loss : float
            Average loss over all batches in this epoch.
        metrics : dict[str, Any] | None, optional
            Additional metrics to track.
        """
        pass

    @abstractmethod
    def on_training_end(self) -> None:
        """Called when training completes."""
        pass
