from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset


class VariabilityMetric(ABC):
    """Abstract interface for variability metrics."""
    
    @abstractmethod
    def compute(self, dataset: Dataset) -> float:
        """
        Compute variability metric for a dataset.
        
        Args:
            dataset: Dataset to compute variability for
            
        Returns:
            Variability score
        """
        pass


class SimilarityMetric(ABC):
    """Abstract interface for similarity metrics."""
    
    @abstractmethod
    def compute(self, real_dataset: Dataset, generated: torch.Tensor) -> float:
        """
        Compute similarity between real and generated data.
        
        Args:
            real_dataset: Real dataset
            generated: Generated images tensor
            
        Returns:
            Similarity score
        """
        pass
