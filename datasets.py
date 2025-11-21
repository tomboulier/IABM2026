from medmnist import ChestMNIST, PathMNIST, DermaMNIST, OCTMNIST
from logging import getLogger

logger = getLogger(__name__)

def load_medmnist(name: str):
    """
    Load a MedMNIST dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset, e.g. "ChestMNIST".

    Returns
    -------
    torch.utils.data.Dataset
        The MedMNIST dataset instance.
    """
    available = {
        "ChestMNIST": ChestMNIST,
        "PathMNIST": PathMNIST,
        "DermaMNIST": DermaMNIST,
        "OCTMNIST": OCTMNIST,
    }

    if name not in available:
        raise ValueError(f"Unknown dataset: {name}")
    
    logger.info(f"Loading {name} dataset...")

    DatasetClass = available[name]
    return DatasetClass(download=True, as_rgb=True, split="train")
