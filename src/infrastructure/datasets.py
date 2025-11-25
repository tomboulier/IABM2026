"""Dataset loading - concrete PyTorch/MedMNIST implementation.

This belongs in infrastructure layer because it depends on PyTorch and MedMNIST.
"""
import torch
import medmnist
from medmnist import INFO
from torchvision import transforms
from src.domain.entities.dataset import Dataset


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_samples)))

    return dataset
