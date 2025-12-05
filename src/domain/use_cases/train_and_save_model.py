"""
Train and Save Model Use-Case.

This module provides a use-case for training a generative model on a dataset
and saving the trained weights to disk for later use.
"""
import logging
from typing import Optional

from src.domain.interfaces.dataset_loader import DatasetLoader
from src.domain.interfaces.model import Model
from src.domain.interfaces.model_handler import ModelHandler

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
    model_handler : Optional[ModelHandler]
        Optional handler for model persistence operations.
        If provided, validates the output path before training and
        uses handler.save() instead of model.save().
    """

    def __init__(
        self,
        dataset_name: str,
        max_samples: int | None,
        image_size: int,
        dataset_loader: DatasetLoader,
        model: Model,
        output_path: str,
        model_handler: Optional[ModelHandler] = None,
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
        model_handler : ModelHandler | None, optional
            Handler for model persistence. If provided, validates the
            output path before training to fail fast on invalid paths.
        """
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.image_size = image_size
        self.dataset_loader = dataset_loader
        self.model = model
        self.output_path = output_path
        self.model_handler = model_handler

    def run(self) -> None:
        """
        Execute the training and saving workflow.

        This method:
        1. Validates the output path (if model_handler is provided)
        2. Loads the dataset using the configured loader
        3. Trains the model on the loaded dataset
        4. Saves the trained model weights to the output path
        """
        # Validate output path BEFORE training if handler is provided
        if self.model_handler is not None:
            logger.info("Validating output path...")
            self.model_handler.validate_save_path(self.output_path)

        logger.info(f"Loading dataset {self.dataset_name}...")
        dataset = self.dataset_loader.load(
            self.dataset_name,
            self.max_samples,
            self.image_size,
        )
        logger.info(f"Loaded {len(dataset)} samples.")

        logger.info(f"Training model on {self.dataset_name}...")
        self.model.train(dataset, dataset_name=self.dataset_name)
        logger.info("Training completed.")

        logger.info(f"Saving model to {self.output_path}...")
        if self.model_handler is not None:
            self.model_handler.save(self.output_path)
        else:
            self.model.save(self.output_path)
        logger.info("Model saved successfully.")
