from abc import ABC, abstractmethod
from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor


class Model(ABC):
    """Abstract interface for generative models."""
    
    @abstractmethod
    def train(self, dataset: Dataset, dataset_name: str | None = None) -> None:
        """
        Train the model using the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to use for training the model.
        dataset_name : str | None, optional
            Name of the dataset (for tracking/logging purposes).
        """
        pass
    
    @abstractmethod
    def generate_images(self, n: int) -> Tensor:
        """
        Generate a batch of images from the trained model.

        Parameters
        ----------
        n : int
            Number of images to generate.

        Returns
        -------
        Tensor
            Tensor containing the generated images.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

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
