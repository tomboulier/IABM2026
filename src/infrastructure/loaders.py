from torch.utils.data import Dataset as TorchDataset
from src.domain.interfaces.dataset_loader import DatasetLoader
from src.domain.entities.datasets import load_dataset, Dataset


class MedMNISTDatasetLoader(DatasetLoader):
    """Concrete implementation for loading MedMNIST datasets."""
    
    def load(self, dataset_name: str, max_samples: int | None, image_size: int) -> Dataset:
        """Load a MedMNIST dataset."""
        return load_dataset(dataset_name, max_samples, image_size)
