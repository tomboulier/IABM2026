"""Dataset loading - concrete PyTorch/MedMNIST implementation.
Loads MedMNIST datasets and adapts them to the Dataset entity.
"""
import medmnist
import numpy
import torch
from medmnist import INFO
from torchvision import transforms

from src.domain.entities.dataset import Dataset


class MedMNISTDatasetAdapter(Dataset):
    """Adapter to make MedMNIST datasets conform to the Dataset entity."""

    def __init__(self, medmnist_dataset):
        self._dataset = medmnist_dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, numpy.ndarray]:
        image, label = self._dataset[idx]
        return image, label

    @property
    def num_channels(self) -> int:
        return self._dataset[0][0].shape[0]  # Assuming image shape is (C, H, W)

    @property
    def image_size(self) -> int:
        # Verify that images are square (H == W)
        if self._dataset[0][0].shape[1] != self._dataset[0][0].shape[2]:
            raise ValueError(f"Images are not square in the dataset {self._dataset}: "
                             f"height {self._dataset[0][0].shape[1]} != width {self._dataset[0][0].shape[2]}")
        return self._dataset[0][0].shape[1]


def load_dataset(name: str, max_samples: int | None = None, image_size: int = 224) -> Dataset:
    """
    Loads a MedMNIST dataset by name.
    
    Args:
        name: The name of the dataset (e.g., 'ChestMNIST').
        max_samples: Optional limit on the number of samples to load.
        image_size: The size to resize images to (default: 224).
        
    Returns:
        A PyTorch Dataset containing the training split.
    """
    name = name.lower()
    if name not in INFO:
        raise ValueError(f"Dataset {name} not found in MedMNIST. Available: {list(INFO.keys())}")

    info = INFO[name]
    DataClass = getattr(medmnist, info['python_class'])

    # Standard transforms for MedMNIST + ImageNet normalization for ResNet compatibility
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load the dataset (download if needed)
    dataset = DataClass(
        split='train',
        transform=data_transform,
        download=True,
        as_rgb=True,
        size=image_size
    )

    # Override transform to include resize if not already image_size
    # Note: medmnist classes handle 'size' param in constructor if supported, 
    # but let's be explicit with transforms to ensure consistency.
    dataset.transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_samples)))

    return MedMNISTDatasetAdapter(dataset)
