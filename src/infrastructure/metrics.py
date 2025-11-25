from src.domain.interfaces.metrics import VariabilityMetric, SimilarityMetric
from src.domain.entities.datasets import Dataset
from src.domain.entities.models import Tensor
import torch
from torch.utils.data import Dataset as TorchDataset
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg

class ResNet:
    """
    Encapsulates the ResNet18 feature extractor logic.
    """
    def __init__(self, device='cpu'):
        """
        Initializes the ResNet18 model with pre-trained weights and removes the classifier.
        """
        self.device = device
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()  # Remove classifier
        model.eval()
        self.model = model.to(self.device)

    def get_backbone(self) -> nn.Module:
        """
        Returns the ResNet backbone model without the classification head.
        """
        return self.model

    def get_features(self, dataset: TorchDataset, batch_size: int = 32) -> np.ndarray:
        """
        Extracts features from a dataset using the encapsulated ResNet model.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        features_list = []

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                # Ensure 3 channels for ResNet
                if inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)

                features = self.model(inputs)
                features_list.append(features.cpu().numpy())

        return np.concatenate(features_list, axis=0)



class ResNetMSDVariabilityMetric(VariabilityMetric):
    """Concrete implementation using ResNet features and mean squared distance to centroid."""
    
    def compute(self, dataset: Dataset) -> float:
        """
        Computes the variability of a dataset as the mean squared distance 
        of feature vectors to their centroid.
        """
        resnet = ResNet()
        features = resnet.get_features(dataset)
        
        centroid = np.mean(features, axis=0)
        diffs = features - centroid
        sq_dists = np.sum(diffs**2, axis=1)
        variability = np.mean(sq_dists)
        
        return float(variability)


class FIDSimilarityMetric(SimilarityMetric):
    """Concrete implementation using Fréchet Inception Distance."""
    
    def compute(self, real_dataset: Dataset, generated: Tensor) -> float:
        """
        Computes the Fréchet Inception Distance (FID) between two datasets.
        Uses the same ResNet18 backbone for consistency in this project context,
        calculating the Fréchet distance between Gaussian distributions fitted to features.
        """
        resnet = ResNet()
        
        # Extract features
        feats_real = resnet.get_features(real_dataset)
        
        # Handle generated data: if it's a tensor of images, wrap it
        if isinstance(generated, torch.Tensor):
            class TensorDataset(torch.utils.data.Dataset):
                def __init__(self, tensor):
                    self.tensor = tensor
                def __getitem__(self, index):
                    return self.tensor[index], 0 # Dummy label
                def __len__(self):
                    return len(self.tensor)
            generated = TensorDataset(generated)
            
        feats_gen = resnet.get_features(generated)
        
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
