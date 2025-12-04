"""
Train and Save Model Use-Case.

This module provides a use-case for training a generative model on a dataset
and saving the trained weights to disk for later use.
"""
import logging

from src.domain.interfaces.dataset_loader import DatasetLoader
from src.domain.interfaces.model import Model

logger = logging.getLogger(__name__)


class TrainAndSaveModel:
    """
    Use-case for training a model on a dataset and saving weights.

    This use-case orchestrates the workflow of:
    1. Loading a dataset by name
    2. Training the model on that dataset
    3. Saving the trained model weights to disk

    Attributes
    ----------
    dataset_name : str
        Name of the dataset to train on.
    max_samples : int | None
        Maximum number of samples to load from the dataset.
    image_size : int
        Target image size for the dataset.
    dataset_loader : DatasetLoader
        Component responsible for loading datasets.
    model : Model
        Model to train and save.
    output_path : str
        Path where the trained model weights will be saved.
    """

    def __init__(
        self,
        dataset_name: str,
        max_samples: int | None,
        image_size: int,
        dataset_loader: DatasetLoader,
        model: Model,
        output_path: str,
    ) -> None:
        """
        Initialize the TrainAndSaveModel use-case.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to train on (e.g., "PathMNIST").
        max_samples : int | None
            Maximum number of samples to load, or None to use all samples.
        image_size : int
            Target image size (pixels) for loading the dataset.
        dataset_loader : DatasetLoader
            Component responsible for loading datasets.
        model : Model
            Model to train and save.
        output_path : str
            Path where the trained model weights will be saved.
        """
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.image_size = image_size
        self.dataset_loader = dataset_loader
        self.model = model
        self.output_path = output_path

    def run(self) -> None:
        """
        Execute the training and saving workflow.

        This method:
        1. Loads the dataset using the configured loader
        2. Trains the model on the loaded dataset
        3. Saves the trained model weights to the output path
        """
        logger.info(f"Loading dataset {self.dataset_name}...")
        dataset = self.dataset_loader.load(
            self.dataset_name,
            self.max_samples,
            self.image_size,
        )
        logger.info(f"Loaded {len(dataset)} samples.")

        logger.info(f"Training model on {self.dataset_name}...")
        self.model.train(dataset)
        logger.info("Training completed.")

        logger.info(f"Saving model to {self.output_path}...")
        self.model.save(self.output_path)
        logger.info("Model saved successfully.")
