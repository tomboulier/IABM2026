"""Model definitions and Tensor protocol."""
from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn


@runtime_checkable
class Tensor(Protocol):
    """Framework-independent tensor/array abstraction."""
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the tensor."""
        ...


class DiffusionModel:
    """
    A minimal diffusion model interface.
    """
    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, dataset):
        """
        Mocks the training process.
        In a real implementation, this would run the diffusion training loop.
        """
        # For this minimal implementation, we just store a reference to a sample
        # to "learn" the shape and statistics for generation.
        self.sample_data, _ = dataset[0]
        self.is_trained = True
        print("Diffusion model 'trained' (mock).")

    def generate_images(self, n: int = 1000) -> torch.Tensor:
        """
        Generates n images.
        Returns a tensor of shape (n, C, H, W).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generation.")
            
        C, H, W = self.sample_data.shape
        
        # Generate random noise for now as a placeholder for diffusion output
        # In a real model, this would be the reverse diffusion process
        generated = torch.randn(n, C, H, W)
        
        # Normalize to roughly match image range [0, 1] or [-1, 1] depending on convention
        # Here we just return the noise
        return generated

# Singleton instance as requested by the API style
diffusion_model = DiffusionModel()
