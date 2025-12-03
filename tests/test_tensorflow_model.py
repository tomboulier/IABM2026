"""Unit tests for TensorFlow diffusion model implementation."""
import pytest
import numpy as np
from src.infrastructure.tensorflow_diffusion_model import TensorFlowDiffusionModelAdapter


class MinimalDataset:
    """
    Minimal dataset for fast testing.
    Creates 10 simple grayscale images.
    """
    def __init__(self, size=28, num_samples=10):
        """
        Initialize the minimal dataset with a fixed number of random grayscale images.
        
        Parameters:
        	size (int): Height and width of each generated square image (default 28).
        	num_samples (int): Number of images to generate (default 10).
        
        Attributes:
        	size (int): The image height/width.
        	num_samples (int): The number of samples in the dataset.
        	images (list[np.ndarray]): List of `num_samples` random float32 arrays shaped (size, size, 1).
        """
        self.size = size
        self.num_samples = num_samples
        # Create simple random images
        self.images = [
            np.random.rand(size, size, 1).astype(np.float32)
            for _ in range(num_samples)
        ]
    
    def __len__(self):
        """
        Return the number of samples available in the dataset.
        
        Returns:
            int: The total number of samples.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Return the dataset item (image and dummy label) at the given index.
        
        Parameters:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: A pair (image, label) where `image` is a float32 NumPy array with shape (size, size, 1) and `label` is the integer `0` (dummy label).
        """
        return self.images[idx], 0  # image, dummy label


def test_tensorflow_model_creation():
    """Test that the TensorFlow model can be created."""
    model = TensorFlowDiffusionModelAdapter(
        image_size=28,
        num_channels=1,
        noise_embedding_size=32,
        batch_size=4,
        epochs=1,
        plot_diffusion_steps=5  # Very few steps for fast testing
    )
    assert model is not None
    assert model.image_size == 28


@pytest.mark.filterwarnings("ignore:Your input ran out of data:UserWarning")
def test_tensorflow_model_training():
    """
    Test that the TensorFlow model trains without errors.
    Uses minimal dataset and parameters for fast execution.
    """
    # Create minimal dataset
    dataset = MinimalDataset(size=28, num_samples=10)
    
    # Create model with minimal parameters
    model = TensorFlowDiffusionModelAdapter(
        image_size=28,
        num_channels=1,
        noise_embedding_size=16,  # Small embedding for speed
        batch_size=5,
        epochs=1,  # Just 1 epoch
        plot_diffusion_steps=3  # Very few diffusion steps
    )
    
    # Train should not raise errors
    try:
        model.train(dataset)
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")


@pytest.mark.filterwarnings("ignore:Your input ran out of data:UserWarning")
def test_tensorflow_model_generation():
    """
    Test that the TensorFlow model generates images with correct shape.
    """
    # Create and train model
    dataset = MinimalDataset(size=28, num_samples=10)
    model = TensorFlowDiffusionModelAdapter(
        image_size=28,
        num_channels=1,
        noise_embedding_size=16,
        batch_size=5,
        epochs=1,
        plot_diffusion_steps=3
    )
    model.train(dataset)
    
    # Generate a few images
    n_generated = 5
    generated = model.generate_images(n=n_generated)
    
    # Check output is numpy array (Tensor protocol)
    assert isinstance(generated, np.ndarray), "Generated output should be numpy array"
    
    # Check shape
    assert generated.shape == (n_generated, 28, 28, 1), \
        f"Expected shape (5, 28, 28, 1), got {generated.shape}"


def test_tensorflow_model_implements_interface():
    """Verify that TensorFlowDiffusionModelAdapter implements Model interface."""
    from src.domain.interfaces.model import Model
    
    model = TensorFlowDiffusionModelAdapter()
    assert isinstance(model, Model), "Model should implement domain Model interface"
    
    # Check required methods exist
    assert hasattr(model, 'train')
    assert hasattr(model, 'generate_images')