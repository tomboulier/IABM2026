from abc import ABC, abstractmethod
from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor


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
    def generate_images(self, n: int) -> Tensor:
        """
        Generate images using the trained model.
        
        Args:
            n: Number of images to generate
            
        Returns:
            Generated images tensor
        """
        pass

