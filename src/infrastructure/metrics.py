from src.domain.interfaces.metrics import VariabilityMetric, SimilarityMetric
from src.domain.entities.metrics import resnet_features_mean_square_centroid, frechet_inception_distance
import torch
from torch.utils.data import Dataset


class ResNetMSDVariabilityMetric(VariabilityMetric):
    """Concrete implementation using ResNet features and mean squared distance to centroid."""
    
    def compute(self, dataset: Dataset) -> float:
        """Compute variability using ResNet features."""
        return resnet_features_mean_square_centroid(dataset)


class FIDSimilarityMetric(SimilarityMetric):
    """Concrete implementation using Fréchet Inception Distance."""
    
    def compute(self, real_dataset: Dataset, generated: torch.Tensor) -> float:
        """Compute FID between real and generated data."""
        return frechet_inception_distance(real_dataset, generated)
