import pytest

from src.domain.use_cases.experiment import Experiment
from src.infrastructure.loaders import MedMNISTDatasetLoader
from src.infrastructure.metrics import ResNetMSDVariabilityMetric, FIDSimilarityMetric


def test_run_experiment(test_datasets, dummy_model):
    """
    Verifies that the experiment runs correctly with a lightweight configuration.
    Uses dependency injection to provide concrete implementations to the use case.
    """
    # Instantiate concrete dependencies (infrastructure layer)
    dataset_loader = MedMNISTDatasetLoader()
    variability_metric = ResNetMSDVariabilityMetric()
    similarity_metric = FIDSimilarityMetric()
    # model is injected via fixture (DummyModel for fast testing)

    # Create experiment (use case - domain layer)
    # Concrete implementations are injected, not created inside use case
    experiment = Experiment(
        datasets=test_datasets["datasets"],
        max_samples=test_datasets["max_samples"],
        image_size=test_datasets["image_size"],
        dataset_loader=dataset_loader,
        variability_metric=variability_metric,
        similarity_metric=similarity_metric,
        model=dummy_model  # Dependency injection
    )

    # Run experiment
    try:
        experiment.run()
    except Exception as e:
        pytest.fail(f"Experiment failed with error: {e}")
