from abc import ABC, abstractmethod
from src.domain.entities.datasets import Dataset


class DatasetLoader(ABC):
    """Abstract interface for dataset loading."""
    
    @abstractmethod
    def load(self, dataset_name: str, max_samples: int | None, image_size: int) -> Dataset:
        """
        Load a dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            max_samples: Maximum number of samples to load (None for all)
            image_size: Size to resize images to
            
        Returns:
            Dataset instance
        """
        pass

