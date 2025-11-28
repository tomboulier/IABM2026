import argparse

from src.domain.use_cases.experiment import Experiment
from src.infrastructure.configuration import ExperimentConfiguration
from src.infrastructure.loaders import MedMNISTDatasetLoader
from src.infrastructure.logging import setup_logging
from src.infrastructure.metrics import FIDSimilarityMetric, ResNetMSDVariabilityMetric
from src.infrastructure.tensorflow.diffusion_model import (
    TensorFlowDiffusionModel,
)


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
    model = TensorFlowDiffusionModel(
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
        model=model,
    )
    experiment.run()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)
