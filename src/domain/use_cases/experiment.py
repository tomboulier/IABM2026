import logging

from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor
from src.domain.interfaces.dataset_loader import DatasetLoader
from src.domain.interfaces.metrics import SimilarityMetric, VariabilityMetric
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
        Create an Experiment configured with datasets, loading/training components, and evaluation metrics.
        
        Parameters:
            datasets (list[str]): Names of datasets to process.
            max_samples (int | None): Maximum number of samples to load per dataset, or None to use all samples.
            image_size (int): Target image size (pixels) used when loading datasets.
            dataset_loader (DatasetLoader): Component responsible for loading datasets.
            variability_metric (VariabilityMetric): Component used to compute dataset variability.
            similarity_metric (SimilarityMetric): Component used to compute similarity between real and generated data.
            model (Model): Model used for training and image generation.
        """
        self.datasets = datasets
        self.max_samples = max_samples
        self.image_size = image_size
        self.dataset_loader = dataset_loader
        self.variability_metric = variability_metric
        self.similarity_metric = similarity_metric
        self.model = model
        
    def load_dataset(self, dataset_name: str) -> Dataset:
        """
        Load a dataset by name using the configured dataset loader.
        
        Parameters:
            dataset_name (str): Name of the dataset to load.
        
        Returns:
            dataset (Dataset): Dataset loaded with this Experiment's `max_samples` limit and `image_size`.
        """
        return self.dataset_loader.load(
            dataset_name,
            self.max_samples,
            self.image_size
        )
    
    def compute_variability(self, dataset: Dataset) -> float:
        """
        Compute the variability of a dataset using the configured variability metric.
        
        Parameters:
            dataset (Dataset): The dataset to evaluate.
        
        Returns:
            float: Variability score for the dataset.
        """
        return self.variability_metric.compute(dataset)
    
    def train_model(self, dataset: Dataset):
        """Train the model on the dataset."""
        self.model.train(dataset)
    
    def generate_images(self, n: int = 100) -> Tensor:
        """
        Generate a batch of images from the experiment's model.
        
        Parameters:
            n (int): Number of images to generate.
        
        Returns:
            Tensor: Tensor containing the generated images.
        """
        return self.model.generate_images(n=n)
    
    def compute_similarity(self, dataset: Dataset, generated: Tensor) -> float:
        """
        Compute similarity between a real dataset and generated images using the configured similarity metric.
        
        Parameters:
            dataset (Dataset): The real dataset to compare against.
            generated (Tensor): Tensor of generated images to compare with the real dataset.
        
        Returns:
            float: Similarity score produced by the configured similarity metric.
        """
        return self.similarity_metric.compute(dataset, generated)
    
    def run(self):
        """
        Orchestrate the full experiment pipeline across configured datasets.
        
        For each dataset name in self.datasets this method loads the dataset, computes its variability, trains the injected model, generates images, and computes similarity between real and generated samples. Progress and key results are logged. If processing a dataset raises an exception, the error is logged for that dataset and the exception is re-raised.
        """
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

            except Exception as error:
                logger.error(f"Failed to process {dataset_name}: {error}")
                raise error
        
        logger.info("Experiment completed.")