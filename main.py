import argparse
import csv
import logging
import os

from src.domain.use_cases.experiment import Experiment, ExperimentResult
from src.infrastructure.configuration import ExperimentConfiguration
from src.infrastructure.loaders import MedMNISTDatasetLoader
from src.infrastructure.logging import setup_logging
from src.infrastructure.metrics import FIDSimilarityMetric, ResNetMSDVariabilityMetric
from src.infrastructure.tensorflow.diffusion_model import TensorFlowDiffusionModel

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run experiment with specified configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="configuration.toml",
        help="Path to experiment configuration TOML file (default: configuration.toml)",
    )
    return parser.parse_args(argv)


def create_model_factory(image_size: int, num_channels: int = 3):
    """
    Create a factory function for TensorFlowDiffusionModel.

    Parameters
    ----------
    image_size : int
        Image size for the model.
    num_channels : int
        Number of image channels.

    Returns
    -------
    Callable[[str | None], Model]
        Factory function that creates a model, optionally loading weights.
    """

    def factory(weights_path: str | None) -> TensorFlowDiffusionModel:
        model = TensorFlowDiffusionModel(
            image_size=image_size,
            num_channels=num_channels,
        )
        if weights_path is not None:
            model.load(weights_path)
        return model

    return factory


def save_results_to_csv(results: list[ExperimentResult], output_path: str) -> None:
    """
    Save experiment results to a CSV file.

    Parameters
    ----------
    results : list[ExperimentResult]
        List of experiment results.
    output_path : str
        Path to the output CSV file.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "dataset",
        "model_weights",
        "variability",
        "similarity",
        "num_samples",
        "num_generated",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "dataset": result.dataset,
                "model_weights": result.model_weights or "",
                "variability": f"{result.variability:.6f}",
                "similarity": f"{result.similarity:.6f}",
                "num_samples": result.num_samples,
                "num_generated": result.num_generated,
            })

    logger.info(f"Results saved to {output_path}")


def main(config_path: str = None):
    # 1. Setup Logging
    setup_logging()

    # 2. Load Configuration
    if config_path is None:
        config_path = "configuration.toml"
    config = ExperimentConfiguration.load(config_path)

    # 3. Instantiate concrete dependencies (outer layer responsibility)
    dataset_loader = MedMNISTDatasetLoader()
    variability_metric = ResNetMSDVariabilityMetric()
    similarity_metric = FIDSimilarityMetric()
    model_factory = create_model_factory(
        image_size=config.image_size,
        num_channels=3,
    )

    # 4. Create and run experiment
    experiment = Experiment(
        datasets=config.datasets,
        max_samples=config.max_samples,
        image_size=config.image_size,
        dataset_loader=dataset_loader,
        variability_metric=variability_metric,
        similarity_metric=similarity_metric,
        model_factory=model_factory,
        weights=config.weights,
        num_generated_images=config.num_generated_images,
    )
    results = experiment.run()

    # 5. Save results to CSV
    save_results_to_csv(results, config.results_output)


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)
