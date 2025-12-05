"""
Generate and Save Images Use-Case.

This module provides a use-case for generating images using a pre-trained
generative model and saving them to disk.
"""
import logging
import os
from typing import Optional

import numpy as np
from PIL import Image

from src.domain.interfaces.model import Model
from src.domain.interfaces.model_handler import ModelHandler

logger = logging.getLogger(__name__)


class GenerateAndSaveImages:
    """
    Use-case for generating images and saving them to disk.

    This use-case orchestrates the workflow of:
    1. Optionally loading model weights from disk
    2. Generating a specified number of images
    3. Saving the generated images as PNG files

    Attributes
    ----------
    model : Model
        The generative model to use for image generation.
    num_images : int
        Number of images to generate.
    output_dir : str
        Directory where generated images will be saved.
    weights_path : str | None
        Optional path to model weights to load before generation.
    model_handler : ModelHandler | None
        Optional handler for model persistence operations.
    """

    def __init__(
        self,
        model: Model,
        num_images: int,
        output_dir: str,
        weights_path: Optional[str] = None,
        model_handler: Optional[ModelHandler] = None,
    ) -> None:
        """
        Initialize the GenerateAndSaveImages use-case.

        Parameters
        ----------
        model : Model
            The generative model to use for image generation.
        num_images : int
            Number of images to generate.
        output_dir : str
            Directory where generated images will be saved.
        weights_path : str | None, optional
            Path to model weights to load before generation.
            If None, the model is assumed to be already loaded/trained.
        model_handler : ModelHandler | None, optional
            Handler for model persistence. If provided, uses handler.load()
            instead of model.load().
        """
        self.model = model
        self.num_images = num_images
        self.output_dir = output_dir
        self.weights_path = weights_path
        self.model_handler = model_handler

    def run(self) -> None:
        """
        Execute the image generation and saving workflow.

        This method:
        1. Creates the output directory if it doesn't exist
        2. Loads model weights if a path was provided
        3. Generates the requested number of images
        4. Saves each image as a PNG file in the output directory
        """
        # Create output directory if needed
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        # Load weights if path provided
        if self.weights_path is not None:
            logger.info(f"Loading model weights from {self.weights_path}...")
            if self.model_handler is not None:
                self.model_handler.load(self.weights_path)
            else:
                self.model.load(self.weights_path)

        # Generate images
        logger.info(f"Generating {self.num_images} images...")
        images = self.model.generate_images(self.num_images)
        logger.info(f"Generated {len(images)} images.")

        # Save images
        logger.info(f"Saving images to {self.output_dir}...")
        for i, img_array in enumerate(images):
            self._save_image(img_array, i)
        logger.info(f"Saved {len(images)} images.")

    def _save_image(self, img_array: np.ndarray, index: int) -> None:
        """
        Save a single image array as a PNG file.

        Parameters
        ----------
        img_array : np.ndarray
            Image array in [0, 1] range with shape (H, W, C).
        index : int
            Index used for the filename.
        """
        # Convert from [0, 1] to [0, 255] and ensure uint8
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)

        # Create PIL Image and save
        img = Image.fromarray(img_array, mode="RGB")
        filename = f"generated_{index:04d}.png"
        filepath = os.path.join(self.output_dir, filename)
        img.save(filepath)
