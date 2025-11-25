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
        self.sample_data = None
        self.is_trained = False

    def train(self, dataset: Dataset):
        """
        Mocks the training process.
        Just stores a reference to understand data shape.
        """
        # Get first sample to understand shape
        self.sample_data, _ = dataset[0]
        self.is_trained = True
        print("DummyModel 'trained' (mock).")

    def generate_images(self, n: int = 1000) -> Tensor:
        """
        Generates n images.
        Returns a numpy array of random noise.
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
