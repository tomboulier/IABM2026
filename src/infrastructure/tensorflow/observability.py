"""
Observability module for TensorFlow training.

This module provides:
- TrainingTracker implementations (ConsoleTracker, SilentTracker)
- TensorFlow logging suppression utilities
- Progress bar integration using tqdm
"""
import logging
import os
from typing import Any

from tqdm import tqdm

from src.domain.interfaces.training_tracker import TrainingTracker


def suppress_tensorflow_logging() -> None:
    """
    Suppress verbose TensorFlow logging messages.

    This redirects TensorFlow's internal logging to Python's logging system
    and sets the log level to ERROR to hide INFO/WARNING messages like:
    - "Local rendezvous is aborting with status: OUT_OF_RANGE"
    - "Loaded cuDNN version..."
    - CPU/GPU feature warnings

    Should be called before importing TensorFlow.
    """
    # Set TensorFlow log level via environment variable (before TF import)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=ALL, 1=WARNING+, 2=ERROR+, 3=FATAL

    # Suppress absl logging (used by TensorFlow internally)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    # Try to suppress TensorFlow's own logger if already imported
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
    except ImportError:
        pass


class ConsoleTracker(TrainingTracker):
    """
    Training tracker with tqdm progress bars and console output.

    Displays:
    - Overall epoch progress bar
    - Per-epoch batch progress bar with live loss display
    - Summary statistics at epoch end

    Attributes
    ----------
    show_batch_loss : bool
        Whether to show loss in the batch progress bar.
    """

    def __init__(self, show_batch_loss: bool = True) -> None:
        """
        Initialize the ConsoleTracker.

        Parameters
        ----------
        show_batch_loss : bool, optional
            Whether to show loss in the batch progress bar. Default is True.
        """
        self.show_batch_loss = show_batch_loss
        self._epoch_pbar: tqdm | None = None
        self._batch_pbar: tqdm | None = None
        self._total_epochs = 0
        self._total_batches = 0
        self._dataset_name: str | None = None
        self._epoch_losses: list[float] = []

    def on_training_start(
        self,
        total_epochs: int,
        total_batches: int,
        dataset_name: str | None = None,
    ) -> None:
        """
        Called when training begins. Initializes the epoch progress bar.

        Parameters
        ----------
        total_epochs : int
            Total number of epochs to train.
        total_batches : int
            Total number of batches per epoch.
        dataset_name : str | None, optional
            Name of the dataset being trained on.
        """
        self._total_epochs = total_epochs
        self._total_batches = total_batches
        self._dataset_name = dataset_name

        desc = "Training"
        if dataset_name:
            desc = f"Training on {dataset_name}"

        self._epoch_pbar = tqdm(
            total=total_epochs,
            desc=desc,
            unit="epoch",
            position=0,
            leave=True,
        )

    def on_epoch_start(self, epoch: int) -> None:
        """
        Called at the start of each epoch. Initializes the batch progress bar.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        """
        self._epoch_losses = []
        self._batch_pbar = tqdm(
            total=self._total_batches,
            desc=f"Epoch {epoch + 1}/{self._total_epochs}",
            unit="batch",
            position=1,
            leave=False,
        )

    def on_batch_end(
        self,
        epoch: int,
        batch: int,
        loss: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Called after each batch. Updates the batch progress bar.

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
        self._epoch_losses.append(loss)

        if self._batch_pbar is not None:
            postfix = {"loss": f"{loss:.4f}"}
            if metrics:
                postfix.update({k: f"{v:.4f}" for k, v in metrics.items()})
            self._batch_pbar.set_postfix(postfix)
            self._batch_pbar.update(1)

    def on_epoch_end(
        self,
        epoch: int,
        avg_loss: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Called at the end of each epoch. Closes batch progress bar and updates epoch bar.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        avg_loss : float
            Average loss over all batches in this epoch.
        metrics : dict[str, Any] | None, optional
            Additional metrics to track.
        """
        if self._batch_pbar is not None:
            self._batch_pbar.close()
            self._batch_pbar = None

        if self._epoch_pbar is not None:
            postfix = {"avg_loss": f"{avg_loss:.4f}"}
            if metrics:
                postfix.update({k: f"{v:.4f}" for k, v in metrics.items()})
            self._epoch_pbar.set_postfix(postfix)
            self._epoch_pbar.update(1)

    def on_training_end(self) -> None:
        """Called when training completes. Closes all progress bars."""
        if self._batch_pbar is not None:
            self._batch_pbar.close()
        if self._epoch_pbar is not None:
            self._epoch_pbar.close()


class SilentTracker(TrainingTracker):
    """
    Training tracker that produces no output.

    Useful for testing or when running in non-interactive environments
    where progress output is not desired.
    """

    def on_training_start(
        self,
        total_epochs: int,
        total_batches: int,
        dataset_name: str | None = None,
    ) -> None:
        """No-op implementation."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """No-op implementation."""
        pass

    def on_batch_end(
        self,
        epoch: int,
        batch: int,
        loss: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """No-op implementation."""
        pass

    def on_epoch_end(
        self,
        epoch: int,
        avg_loss: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """No-op implementation."""
        pass

    def on_training_end(self) -> None:
        """No-op implementation."""
        pass
