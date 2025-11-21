import logging
from typing import Callable
import torch
from torch.utils.data import Dataset
from src.domain.entities.datasets import load_dataset
from src.domain.entities.metrics import resnet_features_mean_square_centroid, frechet_inception_distance
from src.domain.entities.models import diffusion_model

from src.infrastructure.configuration import ExperimentConfiguration

logger = logging.getLogger(__name__)


class Experiment:
    """
    Orchestrates the variability and similarity experiment using dependency injection.
    """
    
    def __init__(
        self,
        config: ExperimentConfiguration,
        dataset_loader: Callable[[str, int | None, int], Dataset],
        variability_metric: Callable[[Dataset], float],
        similarity_metric: Callable[[Dataset, torch.Tensor], float],
        model
    ):
        """
        Initialize the experiment with dependencies.
        
        Args:
            config: Experiment configuration
            dataset_loader: Function to load datasets
            variability_metric: Function to compute variability
            similarity_metric: Function to compute similarity
            model: Model instance for training and generation
        """
        self.config = config
        self.dataset_loader = dataset_loader
        self.variability_metric = variability_metric
        self.similarity_metric = similarity_metric
        self.model = model
        
    def load_dataset(self, dataset_name: str) -> Dataset:
        """Load a dataset using the injected loader."""
        return self.dataset_loader(
            dataset_name,
            self.config.max_samples,
            self.config.image_size
        )
    
    def compute_variability(self, dataset: Dataset) -> float:
        """Compute variability using the injected metric."""
        return self.variability_metric(dataset)
    
    def train_model(self, dataset: Dataset):
        """Train the model on the dataset."""
        self.model.train(dataset)
    
    def generate_images(self, n: int = 100) -> torch.Tensor:
        """Generate images using the trained model."""
        return self.model.generate_images(n=n)
    
    def compute_similarity(self, dataset: Dataset, generated: torch.Tensor) -> float:
        """Compute similarity between real and generated data."""
        return self.similarity_metric(dataset, generated)
    
    def run(self):
        """Run the complete experiment workflow."""
        logger.info("Starting MedMNIST Variability & Similarity Experiment")
        
        for dataset_name in self.config.datasets:
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


def run_experiment(config: ExperimentConfiguration):
    """
    Runs the variability and similarity experiment based on the configuration.
    
    Args:
        config: The experiment configuration.
    """
    # Create experiment with default dependencies
    experiment = Experiment(
        config=config,
        dataset_loader=load_dataset,
        variability_metric=resnet_features_mean_square_centroid,
        similarity_metric=frechet_inception_distance,
        model=diffusion_model
    )
    
    # Run the experiment
    experiment.run()
