import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torch import nn
from tqdm import tqdm

from logging import getLogger

logger = getLogger(__name__)

def _build_internal_feature_extractor():
    """
    Internal utility:
    Create a pretrained ResNet18 without the classifier.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    return feature_extractor


def extract_features(dataset):
    """
    Extract a (N, 512) feature matrix from a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 512), CPU.
    """
    logger.info("Extracting features using ResNet18 backbone...")
    # Internal choices (YAGNI: we fix them, not configurable)
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Wrap dataset with transform
    dataset.transform = transform

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = _build_internal_feature_extractor().to(device)
    all_features = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images)                 # (B, 512, 1, 1)
            feats = feats.view(feats.size(0), -1) # (B, 512)
            all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)  # (N, 512)
