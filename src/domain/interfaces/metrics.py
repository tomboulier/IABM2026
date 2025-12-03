from abc import ABC, abstractmethod
from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor


class VariabilityMetric(ABC):
    """Abstract interface for variability metrics."""
    
    @abstractmethod
    def compute(self, dataset: Dataset) -> float:
        """
        Compute a scalar variability score for the given dataset.
        
        Parameters:
            dataset (Dataset): The dataset to evaluate.
        
        Returns:
            float: Variability score where larger values indicate greater variability.
        """
        pass


class SimilarityMetric(ABC):
    """Abstract interface for similarity metrics."""
    
    @abstractmethod
    def compute(self, real_dataset: Dataset, generated: Tensor) -> float:
        """
        Compute how similar generated data is to a real dataset.
        
        Parameters:
            real_dataset (Dataset): The real dataset to compare against.
            generated (Tensor): The generated data tensor to evaluate.
        
        Returns:
            float: Similarity score; higher values indicate greater similarity.
        """
        pass
