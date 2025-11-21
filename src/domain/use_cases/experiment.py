import logging
import torch
from torch.utils.data import Dataset
from src.domain.interfaces.dataset_loader import DatasetLoader
from src.domain.interfaces.metrics import VariabilityMetric, SimilarityMetric
from src.domain.interfaces.model import Model

logger = logging.getLogger(__name__)


class Experiment:
    """
    Orchestrates the variability and similarity experiment using dependency injection.
    """
    
    def __init__(
        self,
        datasets: list[str],
        max_samples: int | None,
        image_size: int,
        dataset_loader: DatasetLoader,
        variability_metric: VariabilityMetric,
        similarity_metric: SimilarityMetric,
        model: Model
    ):
        """
        Initialize the experiment with dependencies.
        
        Args:
            datasets: List of dataset names to process
            max_samples: Maximum samples per dataset
            image_size: Image size for datasets
            dataset_loader: Dataset loader implementation
            variability_metric: Variability metric implementation
            similarity_metric: Similarity metric implementation
            model: Model implementation
        """
        self.datasets = datasets
        self.max_samples = max_samples
        self.image_size = image_size
        self.dataset_loader = dataset_loader
        self.variability_metric = variability_metric
        self.similarity_metric = similarity_metric
        self.model = model
        
    def load_dataset(self, dataset_name: str) -> Dataset:
        """Load a dataset using the injected loader."""
        return self.dataset_loader.load(
            dataset_name,
            self.max_samples,
            self.image_size
        )
    
    def compute_variability(self, dataset: Dataset) -> float:
        """Compute variability using the injected metric."""
        return self.variability_metric.compute(dataset)
    
    def train_model(self, dataset: Dataset):
        """Train the model on the dataset."""
        self.model.train(dataset)
    
    def generate_images(self, n: int = 100) -> torch.Tensor:
        """Generate images using the trained model."""
        return self.model.generate_images(n=n)
    
    def compute_similarity(self, dataset: Dataset, generated: torch.Tensor) -> float:
        """Compute similarity between real and generated data."""
        return self.similarity_metric.compute(dataset, generated)
    
    def run(self):
        """Run the complete experiment workflow."""
        logger.info("Starting MedMNIST Variability & Similarity Experiment")
        
        for dataset_name in self.datasets:
            logger.info(f"Processing {dataset_name}...")
            
            try:
                # 1. Load Dataset
                dataset = self.load_dataset(dataset_name)
                logger.info(f"Loaded {dataset_name} with {len(dataset)} samples.")
                
                # 2. Compute Variability
                variability_score = self.compute_variability(dataset)
                logger.info(f"Variability (Mean Squared Distance to Centroid) for {dataset_name}: {variability_score:.4f}")
                
                # 3. Train Model
                logger.info(f"Training diffusion model on {dataset_name}...")
                self.train_model(dataset)
                
                # 4. Generate Images
                logger.info(f"Generating images for {dataset_name}...")
                generated_images = self.generate_images(n=100)
                
                # 5. Compute Similarity
                similarity_score = self.compute_similarity(dataset, generated_images)
                logger.info(f"Similarity (FID-like) for {dataset_name}: {similarity_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
        
        logger.info("Experiment completed.")

