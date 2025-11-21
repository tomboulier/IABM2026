from datasets import load_medmnist
from features import extract_features
from variability import compute_variability
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():

    dataset_names = [
        "ChestMNIST",
        "PathMNIST",
        "DermaMNIST",
        "OCTMNIST"
    ]

    for name in dataset_names:
        dataset = load_medmnist(name)
        feature_vectors = extract_features(dataset)
        variability = compute_variability(feature_vectors)
        logger.info(f"{name}: {variability:.4f}")


if __name__ == "__main__":
    main()
