"""
Model Handler Interface.

This module defines the abstract interface for model persistence operations.
The ModelHandler is responsible for saving and loading model weights,
and validating file paths before operations.
"""
from abc import ABC, abstractmethod


class ModelHandler(ABC):
    """
    Abstract interface for model persistence operations.

    This interface decouples persistence operations from the model itself,
    allowing different storage backends and formats to be implemented
    while keeping the domain layer framework-agnostic.

    Implementations should validate file paths according to their
    specific format requirements (e.g., `.weights.h5` for Keras,
    `.pth` for PyTorch).
    """

    @abstractmethod
    def validate_save_path(self, path: str) -> None:
        """
        Validate that the save path is acceptable for this handler.

        This method should be called BEFORE any long-running operations
        (like training) to fail fast if the path is invalid.

        Parameters
        ----------
        path : str
            Path where the model weights will be saved.

        Raises
        ------
        ValueError
            If the path is invalid for this handler's format requirements.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model weights to disk.

        Parameters
        ----------
        path : str
            Path where the model weights will be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model weights from disk.

        Parameters
        ----------
        path : str
            Path to the saved model weights file.
        """
        pass
