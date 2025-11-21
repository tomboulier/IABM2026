from src.infrastructure.logging import setup_logging
from src.infrastructure.configuration import ExperimentConfiguration
from src.infrastructure.loaders import MedMNISTDatasetLoader
from src.infrastructure.metrics import ResNetMSDVariabilityMetric, FIDSimilarityMetric
from src.infrastructure.models import DiffusionModelWrapper
from src.domain.use_cases.experiment import Experiment


def main():
    # 1. Setup Logging
    setup_logging()
    
    # 2. Load Configuration
    config = ExperimentConfiguration.load("configuration.toml")
    
    # 3. Instantiate concrete dependencies (outer layer responsibility)
    dataset_loader = MedMNISTDatasetLoader()
    variability_metric = ResNetMSDVariabilityMetric()
    similarity_metric = FIDSimilarityMetric()
    model = DiffusionModelWrapper()
    
    # 4. Create and run experiment
    experiment = Experiment(
        datasets=config.datasets,
        max_samples=config.max_samples,
        image_size=config.image_size,
        dataset_loader=dataset_loader,
        variability_metric=variability_metric,
        similarity_metric=similarity_metric,
        model=model
    )
    experiment.run()


if __name__ == "__main__":
    main()

