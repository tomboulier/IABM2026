import torch
from logging import getLogger

logger = getLogger(__name__)

def compute_variability(feature_vectors: torch.Tensor) -> float:
    """
    Compute the variability score for a dataset based on
    its feature vectors (N, D).

    Parameters
    ----------
    feature_vectors : torch.Tensor

    Returns
    -------
    float
        Mean squared distance to the centroid.
    """
    logger.info("Computing variability...")
    centroid = feature_vectors.mean(dim=0, keepdim=True)
    distances = (feature_vectors - centroid).pow(2).sum(dim=1)
    return distances.mean().item()
