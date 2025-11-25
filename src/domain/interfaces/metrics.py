from abc import ABC, abstractmethod
from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor


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
    def compute(self, real_dataset: Dataset, generated: Tensor) -> float:
        """
        Compute similarity between real and generated data.
        
        Args:
            real_dataset: Real dataset
            generated: Generated images tensor
            
        Returns:
            Similarity score
        """
        pass

