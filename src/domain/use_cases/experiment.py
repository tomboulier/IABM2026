import logging
from dataclasses import dataclass
from typing import Callable

from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor
from src.domain.interfaces.dataset_loader import DatasetLoader
from src.domain.interfaces.metrics import SimilarityMetric, VariabilityMetric
from src.domain.interfaces.model import Model

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single dataset experiment."""

    dataset: str
    model_weights: str | None
    variability: float
    similarity: float
    num_samples: int
    num_generated: int


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
        model_factory: Callable[[str | None], Model],
        weights: dict[str, str] | None = None,
        num_generated_images: int = 100,
    ):
        """
        Create an Experiment configured with datasets, loading/training components, and evaluation metrics.

        Parameters
        ----------
        datasets : list[str]
            Names of datasets to process.
        max_samples : int | None
            Maximum number of samples to load per dataset, or None to use all samples.
        image_size : int
            Target image size (pixels) used when loading datasets.
        dataset_loader : DatasetLoader
            Component responsible for loading datasets.
        variability_metric : VariabilityMetric
            Component used to compute dataset variability.
        similarity_metric : SimilarityMetric
            Component used to compute similarity between real and generated data.
        model_factory : Callable[[str | None], Model]
            Factory function that creates a Model, optionally loading weights from a path.
        weights : dict[str, str] | None
            Mapping of dataset name to weights file path. If None, models will be trained.
        num_generated_images : int
            Number of images to generate for similarity computation.
        """
        self.datasets = datasets
        self.max_samples = max_samples
        self.image_size = image_size
        self.dataset_loader = dataset_loader
        self.variability_metric = variability_metric
        self.similarity_metric = similarity_metric
        self.model_factory = model_factory
        self.weights = weights or {}
        self.num_generated_images = num_generated_images
        
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
    
    def compute_similarity(self, dataset: Dataset, generated: Tensor) -> float:
        """
        Compute similarity between a real dataset and generated images using the configured similarity metric.

        Parameters
        ----------
        dataset : Dataset
            The real dataset to compare against.
        generated : Tensor
            Tensor of generated images to compare with the real dataset.

        Returns
        -------
        float
            Similarity score produced by the configured similarity metric.
        """
        return self.similarity_metric.compute(dataset, generated)

    def run(self) -> list[ExperimentResult]:
        """
        Orchestrate the full experiment pipeline across configured datasets.

        For each dataset, this method:
        1. Loads the dataset
        2. Computes its variability
        3. Loads pre-trained model weights (or trains if no weights provided)
        4. Generates images
        5. Computes similarity between real and generated samples

        Returns
        -------
        list[ExperimentResult]
            Results for each dataset processed.
        """
        logger.info("Starting MedMNIST Variability & Similarity Experiment")
        results: list[ExperimentResult] = []

        for dataset_name in self.datasets:
            logger.info(f"Processing {dataset_name}...")

            try:
                # 1. Load Dataset
                dataset = self.load_dataset(dataset_name)
                logger.info(f"Loaded {dataset_name} with {len(dataset)} samples.")

                # 2. Compute Variability
                variability_score = self.compute_variability(dataset)
                logger.info(
                    f"Variability (Mean Squared Distance to Centroid) for "
                    f"{dataset_name}: {variability_score:.4f}"
                )

                # 3. Create model and load weights (or train)
                weights_path = self.weights.get(dataset_name)
                if weights_path:
                    logger.info(f"Loading pre-trained weights from {weights_path}")
                    model = self.model_factory(weights_path)
                else:
                    logger.info(f"No weights provided, training model on {dataset_name}...")
                    model = self.model_factory(None)
                    model.train(dataset, dataset_name=dataset_name)

                # 4. Generate Images
                logger.info(
                    f"Generating {self.num_generated_images} images for {dataset_name}..."
                )
                generated_images = model.generate_images(n=self.num_generated_images)

                # 5. Compute Similarity
                similarity_score = self.compute_similarity(dataset, generated_images)
                logger.info(f"Similarity (FID) for {dataset_name}: {similarity_score:.4f}")

                # Store result
                result = ExperimentResult(
                    dataset=dataset_name,
                    model_weights=weights_path,
                    variability=variability_score,
                    similarity=similarity_score,
                    num_samples=len(dataset),
                    num_generated=self.num_generated_images,
                )
                results.append(result)

            except Exception as error:
                logger.error(f"Failed to process {dataset_name}: {error}")
                raise error

        logger.info("Experiment completed.")
        return results