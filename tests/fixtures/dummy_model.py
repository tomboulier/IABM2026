"""DummyModel - simple, fast model implementation for testing.

This is a lightweight mock that doesn't do real training or generation,
used to keep tests fast.
"""
import numpy as np
from src.domain.interfaces.model import Model
from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor


class DummyModel(Model):
    """
    A minimal diffusion model for testing.
    Doesn't do real training - just generates random noise.
    """
    def __init__(self):
        """
        Initialize DummyModel internal state.
        
        Sets `sample_data` to None (will hold the first sample from a dataset after training) and `is_trained` to False (indicates whether the model has been mock-trained).
        """
        self.sample_data = None
        self.is_trained = False

    def train(self, dataset: Dataset):
        """
        Mock-trains the model by recording the first sample from the dataset.
        
        Stores the dataset's first sample in self.sample_data, sets self.is_trained to True, and prints a confirmation message.
        
        Parameters:
            dataset (Dataset): Source of training samples; the function reads dataset[0] and stores its first element (typically the input sample).
        """
        # Get first sample to understand shape
        self.sample_data, _ = dataset[0]
        self.is_trained = True
        print("DummyModel 'trained' (mock).")

    def generate_images(self, n: int = 1000) -> Tensor:
        """
        Generate a batch of synthetic images as random Gaussian noise using the training sample's shape.
        
        Parameters:
            n (int): Number of images to generate (default 1000).
        
        Returns:
            Tensor: NumPy array of random values. If the stored training sample has a `shape` attribute, that shape is used; when the shape has length 3 it is interpreted as (C, H, W) and the returned array has shape `(n, C, H, W)`, otherwise the returned array has shape `(n, *shape)`. If no sample shape is available, the default sample shape `(28, 28, 1)` is used.
        
        Raises:
            RuntimeError: If the model has not been trained prior to generation.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generation.")
            
        # Get shape from sample data
        if hasattr(self.sample_data, 'shape'):
            shape = self.sample_data.shape
        else:
            # Fallback if sample_data is not numpy/tensor
            shape = (28, 28, 1)  # default shape
        
        # Generate random noise
        if len(shape) == 3:
            C, H, W = shape
            generated = np.random.randn(n, C, H, W)
        else:
            generated = np.random.randn(n, *shape)
        
        return generated