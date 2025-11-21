from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset


class Model(ABC):
    """Abstract interface for generative models."""
    
    @abstractmethod
    def train(self, dataset: Dataset):
        """
        Train the model on a dataset.
        
        Args:
            dataset: Training dataset
        """
        pass
    
    @abstractmethod
    def generate_images(self, n: int) -> torch.Tensor:
        """
        Generate images using the trained model.
        
        Args:
            n: Number of images to generate
            
        Returns:
            Generated images tensor
        """
        pass
