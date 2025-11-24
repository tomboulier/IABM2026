from src.domain.interfaces.metrics import VariabilityMetric, SimilarityMetric
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg

def _get_backbone():
    """
    Internal helper to get the feature extractor backbone.
    Returns a ResNet18 model with the classifier removed.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity() # Remove classifier
    model.eval()
    return model

def _get_features(dataset, model, batch_size=32, device='cpu'):
    """
    Internal helper to extract features from a dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features_list = []
    
    model = model.to(device)
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            # Ensure 3 channels
            if inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 3, 1, 1)
            
            features = model(inputs)
            features_list.append(features.cpu().numpy())
            
    return np.concatenate(features_list, axis=0)      



class ResNetMSDVariabilityMetric(VariabilityMetric):
    """Concrete implementation using ResNet features and mean squared distance to centroid."""
    
    def compute(self, dataset: Dataset) -> float:
        """
        Computes the variability of a dataset as the mean squared distance 
        of feature vectors to their centroid.
        """
        model = _get_backbone()
        features = _get_features(dataset, model)
        
        centroid = np.mean(features, axis=0)
        diffs = features - centroid
        sq_dists = np.sum(diffs**2, axis=1)
        variability = np.mean(sq_dists)
        
        return float(variability)


class FIDSimilarityMetric(SimilarityMetric):
    """Concrete implementation using Fréchet Inception Distance."""
    
    def compute(self, real_dataset: Dataset, generated: torch.Tensor) -> float:
        """
        Computes the Fréchet Inception Distance (FID) between two datasets.
        Uses the same ResNet18 backbone for consistency in this project context,
        calculating the Fréchet distance between Gaussian distributions fitted to features.
        """
        model = _get_backbone()
        
        # Extract features
        feats_real = _get_features(dataset_real, model)
        
        # Handle generated data: if it's already a tensor of images, wrap it
        if isinstance(dataset_generated, torch.Tensor):
            class TensorDataset(torch.utils.data.Dataset):
                def __init__(self, tensor):
                    self.tensor = tensor
                def __getitem__(self, index):
                    return self.tensor[index], 0 # Dummy label
                def __len__(self):
                    return len(self.tensor)
            dataset_generated = TensorDataset(dataset_generated)
            
        feats_gen = _get_features(dataset_generated, model)
        
        # Calculate statistics
        mu1, sigma1 = np.mean(feats_real, axis=0), np.cov(feats_real, rowvar=False)
        mu2, sigma2 = np.mean(feats_gen, axis=0), np.cov(feats_gen, rowvar=False)
        
        # Calculate Fréchet distance
        ssdiff = np.sum((mu1 - mu2)**2)
        
        # Product of covariances
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Check for numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        
        return float(fid)
