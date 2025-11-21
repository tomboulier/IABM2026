import torch
import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import Dataset

def load_dataset(name: str) -> Dataset:
    """
    Loads a MedMNIST dataset by name.
    
    Args:
        name: The name of the dataset (e.g., 'ChestMNIST').
        
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
        size=224 # Resize to 224 for ResNet compatibility if needed, though MedMNIST is usually 28x28 or 224x224. 
                 # ResNet expects 224 usually. MedMNIST default is 28.
                 # Let's force 224 to be safe for the backbone.
    )
    
    # Override transform to include resize if not already 224
    # Note: medmnist classes handle 'size' param in constructor if supported, 
    # but let's be explicit with transforms to ensure consistency.
    dataset.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    return dataset
