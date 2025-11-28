"""Unit tests for TensorFlow diffusion model implementation."""
import numpy as np
import pytest

from src.domain.entities.dataset import Dataset
from src.infrastructure.tensorflow.diffusion_model import TensorFlowDiffusionModelAdapter


class MinimalDataset:
    """
    Minimal dataset for fast testing.
    Creates 10 simple grayscale images.
    """

    def __init__(self, size=28, num_samples=10):
        self.size = size
        self.num_samples = num_samples
        # Create simple random images
        self.images = [
            np.random.rand(size, size, 3).astype(np.float32)
            for _ in range(num_samples)
        ]
        self.image_size = size
        self.num_channels = self.images[0].shape[-1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], 0  # image, dummy label


class TestTensorFlowModel:
    """Test class for TensorFlow model."""

    @pytest.fixture()
    def minimal_model(self) -> TensorFlowDiffusionModelAdapter:
        # Create model with minimal parameters
        model = TensorFlowDiffusionModelAdapter(
            image_size=28,
            num_channels=3,
            noise_embedding_size=16,  # Small embedding for speed
            batch_size=5,
            epochs=1,  # Just 1 epoch
            plot_diffusion_steps=3  # Very few diffusion steps
        )
        return model

    @pytest.fixture()
    def minimal_dataset(self) -> Dataset:
        return MinimalDataset(size=28, num_samples=10)

    def test_tensorflow_model_creation(self):
        """Test that the TensorFlow model can be created."""
        model = TensorFlowDiffusionModelAdapter(
            image_size=28,
            num_channels=3,
            noise_embedding_size=32,
            batch_size=4,
            epochs=1,
            plot_diffusion_steps=5  # Very few steps for fast testing
        )
        assert model is not None
        assert model.image_size == 28

    @pytest.mark.filterwarnings("ignore:Your input ran out of data:UserWarning")
    def test_tensorflow_model_training(self, minimal_model, minimal_dataset):
        """
        Test that the TensorFlow model trains without errors.
        Uses minimal dataset and parameters for fast execution.
        """
        # Train should not raise errors
        try:
            minimal_model.train(minimal_dataset)
        except Exception as e:
            pytest.fail(f"Training failed with error: {e}")

    @pytest.mark.filterwarnings("ignore:Your input ran out of data:UserWarning")
    def test_tensorflow_model_generation(self, minimal_model, minimal_dataset):
        """
        Test that the TensorFlow model generates images with correct shape.
        """
        # Create and train model
        dataset = MinimalDataset(size=28, num_samples=10)
        minimal_model.train(minimal_dataset)

        # Generate a few images
        n_generated = 5
        generated = minimal_model.generate_images(n=n_generated)

        # Check output is numpy array (Tensor protocol)
        assert isinstance(generated, np.ndarray), "Generated output should be numpy array"

        # Check shape
        expected_shape = (n_generated, minimal_model.image_size, minimal_model.image_size, minimal_model.num_channels)
        assert generated.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {generated.shape}"

    def test_tensorflow_model_implements_interface(self):
        """Verify that TensorFlowDiffusionModelAdapter implements Model interface."""
        from src.domain.interfaces.model import Model

        model = TensorFlowDiffusionModelAdapter()
        assert isinstance(model, Model), "Model should implement domain Model interface"

        # Check required methods exist
        assert hasattr(model, 'train')
        assert hasattr(model, 'generate_images')
