from abc import ABC, abstractmethod
from src.domain.entities.dataset import Dataset


class DatasetLoader(ABC):
    """Abstract interface for dataset loading."""
    
    @abstractmethod
    def load(self, dataset_name: str, max_samples: int | None, image_size: int) -> Dataset:
        """
        Load the specified dataset and return it as a Dataset.
        
        Parameters:
            dataset_name (str): Identifier of the dataset to load.
            max_samples (int | None): Maximum number of samples to include; if None, include all available samples.
            image_size (int): Target size in pixels to which images will be resized.
        
        Returns:
            Dataset: Loaded dataset with images resized to `image_size`.
        """
        pass
