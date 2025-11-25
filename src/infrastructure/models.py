from src.domain.interfaces.model import Model
from src.domain.entities.models import diffusion_model, DiffusionModel, Tensor
from src.domain.entities.datasets import Dataset
import torch
from torch.utils.data import Dataset as TorchDataset


class DiffusionModelWrapper(Model):
    """Concrete implementation wrapping the diffusion model singleton."""
    
    def __init__(self, model_instance: DiffusionModel = None):
        """Initialize with a model instance (defaults to singleton)."""
        self.model_instance = model_instance if model_instance is not None else diffusion_model
    
    def train(self, dataset: Dataset):
        """Train the diffusion model."""
        self.model_instance.train(dataset)
    
    def generate_images(self, n: int) -> Tensor:
        """Generate images using the diffusion model."""
        return self.model_instance.generate_images(n)

