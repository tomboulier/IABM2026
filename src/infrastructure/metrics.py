import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision.models import ResNet18_Weights, resnet18

from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor
from src.domain.interfaces.metrics import SimilarityMetric, VariabilityMetric


class ResNet:
    """
    Encapsulates the ResNet18 feature extractor logic.
    """
    def __init__(self, device='cpu'):
        """
        Create a ResNet-18 feature extractor using ImageNet pretrained weights and remove its classifier.
        
        Initializes a torchvision ResNet-18 with ImageNet weights, replaces the final fully connected layer with an identity module (so the model outputs features instead of class logits), sets the model to evaluation mode, and moves it to the specified device.
        
        Parameters:
            device (str | torch.device): Target device for the model (for example 'cpu' or 'cuda').
        """
        self.device = device
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()  # Remove classifier
        model.eval()
        self.model = model.to(self.device)

    def get_backbone(self) -> nn.Module:
        """
        Get the ResNet backbone model without the classification head.
        
        Returns:
            model (nn.Module): The pretrained ResNet-18 feature extractor with the final fully connected layer replaced (classifier removed).
        """
        return self.model

    def get_features(self, dataset: TorchDataset, batch_size: int = 32) -> np.ndarray:
        """
        Extract feature vectors for all images in a Torch dataset using the ResNet backbone.
        
        Processes the dataset in batches, accepts datasets that yield either (inputs, labels) or inputs only, and handles common input layouts:
        - Adds a batch dimension for single images (3D tensors).
        - Detects and permutes NHWC (batch, H, W, C) to NCHW when appropriate.
        - Repeats a single channel into 3 channels to match ResNet's expected input.
        
        Parameters:
            dataset (torch.utils.data.Dataset): A Torch dataset yielding image tensors or arrays (optionally paired with labels).
            batch_size (int): Number of samples per batch during feature extraction.
        
        Returns:
            np.ndarray: Array of shape (N, D) where N is the total number of samples and D is the backbone feature dimensionality.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        features_list = []

        with torch.no_grad():
            for batch in loader:
                # Support datasets returning either (inputs, labels) or inputs only
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                # Convert numpy arrays to torch tensors if necessary
                if not isinstance(inputs, torch.Tensor):
                    np_inputs = np.array(inputs)
                    inputs = torch.as_tensor(np_inputs, dtype=torch.float32)

                inputs = inputs.to(self.device)

                # Ensure 3 channels for ResNet (repeat single channel into 3)
                # If inputs are NHWC (batch, H, W, C), detect and permute to NCHW
                if inputs.dim() == 3:
                    # single image without batch dim -> add batch
                    inputs = inputs.unsqueeze(0)

                if inputs.dim() == 4:
                    # Heuristic: if channel dim is last (NHWC) and equals 1 or 3, permute
                    if inputs.shape[-1] in (1, 3) and inputs.shape[1] not in (1, 3):
                        inputs = inputs.permute(0, 3, 1, 2).contiguous()

                # Now ensure we have channels-first and repeat single channel into 3
                if inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)

                features = self.model(inputs)
                features_list.append(features.cpu().numpy())

        if len(features_list) == 0:
            return np.zeros((0, self.model.fc.in_features if hasattr(self.model, 'fc') else 512))

        return np.concatenate(features_list, axis=0)



class ResNetMSDVariabilityMetric(VariabilityMetric):
    """Concrete implementation using ResNet features and mean squared distance to centroid."""
    
    def compute(self, dataset: Dataset) -> float:
        """
        Compute the dataset variability as the mean squared Euclidean distance of ResNet feature vectors to their centroid.
        
        Returns:
            variability (float): Mean of squared Euclidean distances from each feature vector to the feature centroid.
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
        Compute the Fréchet Inception Distance (FID) between a real dataset and generated images.
        
        Parameters:
            real_dataset (Dataset): Dataset of real images used as the reference distribution.
            generated (Tensor or Dataset): Either a tensor of images (N,H,W,C or N,C,H,W) or a Dataset producing images; if a tensor is provided it will be wrapped into a simple Dataset.
        
        Returns:
            float: The Fréchet distance between the feature distributions of the two sets; lower values indicate more similar distributions.
        """
        resnet = ResNet()
        
        # Extract features
        feats_real = resnet.get_features(real_dataset)
        
        # Handle generated data: if it's a tensor of images, wrap it
        if isinstance(generated, torch.Tensor):
            class TensorDataset(torch.utils.data.Dataset):
                def __init__(self, tensor):
                    """
                    Initialize the instance with the provided tensor as its primary data.
                    
                    Parameters:
                        tensor (Tensor): The tensor to store on the instance.
                    """
                    self.tensor = tensor
                def __getitem__(self, index):
                    """
                    Retrieve an item from the underlying tensor dataset by index.
                    
                    Parameters:
                        index (int or slice): Index or slice used to select element(s) from the tensor.
                    
                    Returns:
                        tuple: A pair (sample, 0) where `sample` is the tensor element(s) selected by `index` and `0` is a dummy label.
                    """
                    return self.tensor[index], 0 # Dummy label
                def __len__(self):
                    """
                    Get the number of elements (samples) in the wrapped tensor.
                    
                    Returns:
                        int: Number of items in self.tensor.
                    """
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