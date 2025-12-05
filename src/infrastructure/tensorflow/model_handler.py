"""
TensorFlow Model Handler Implementation.

This module provides a TensorFlow/Keras-specific implementation of the
ModelHandler interface, handling weight persistence in the .weights.h5 format.
"""
from src.domain.interfaces.model import Model
from src.domain.interfaces.model_handler import ModelHandler


class TensorFlowModelHandler(ModelHandler):
    """
    TensorFlow/Keras implementation of the ModelHandler interface.

    This handler manages model weight persistence using the Keras
    `.weights.h5` format. It validates paths before operations to
    ensure compatibility with Keras save/load methods.

    Parameters
    ----------
    model : Model
        The model whose weights will be saved/loaded.
    """

    REQUIRED_EXTENSION = ".weights.h5"

    def __init__(self, model: Model) -> None:
        """
        Initialize the TensorFlow model handler.

        Parameters
        ----------
        model : Model
            The model whose weights will be saved/loaded.
        """
        self.model = model

    def validate_save_path(self, path: str) -> None:
        """
        Validate that the save path ends with '.weights.h5'.

        Parameters
        ----------
        path : str
            Path where the model weights will be saved.

        Raises
        ------
        ValueError
            If the path does not end with '.weights.h5'.
        """
        if not path.endswith(self.REQUIRED_EXTENSION):
            raise ValueError(
                f"Weights path must end with '{self.REQUIRED_EXTENSION}'. "
                f"Got: '{path}'"
            )

    def save(self, path: str) -> None:
        """
        Save model weights to disk in Keras .weights.h5 format.

        Parameters
        ----------
        path : str
            Path where the model weights will be saved.
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Load model weights from disk.

        Parameters
        ----------
        path : str
            Path to the saved model weights file.

        Raises
        ------
        ValueError
            If the path does not end with '.weights.h5'.
        """
        self.validate_save_path(path)
        self.model.load(path)
