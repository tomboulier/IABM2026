from torch.utils.data import Dataset as TorchDataset
from src.domain.interfaces.dataset_loader import DatasetLoader
from src.infrastructure.datasets import load_dataset
from src.domain.entities.dataset import Dataset


class MedMNISTDatasetLoader(DatasetLoader):
    """Concrete implementation for loading MedMNIST datasets."""
    
    def load(self, dataset_name: str, max_samples: int | None, image_size: int) -> Dataset:
        """
        Load the specified MedMNIST dataset and return it as a Dataset.
        
        Parameters:
            dataset_name (str): Name of the MedMNIST dataset to load.
            max_samples (int | None): Maximum number of samples to load; None to load all available samples.
            image_size (int): Target image size for loaded samples.
        
        Returns:
            Dataset: Loaded dataset configured with the requested sample limit and image size.
        """
        return load_dataset(dataset_name, max_samples, image_size)