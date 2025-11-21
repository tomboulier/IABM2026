import pytest
import tomli_w
from src.infrastructure.loaders import MedMNISTDatasetLoader
from src.infrastructure.metrics import ResNetMSDVariabilityMetric, FIDSimilarityMetric
from src.infrastructure.models import DiffusionModelWrapper
from src.domain.use_cases.experiment import Experiment


@pytest.fixture
def test_datasets():
    """Lightweight test configuration."""
    return {
        "datasets": ["ChestMNIST"],
        "max_samples": 100,
        "image_size": 28
    }


def test_run_experiment(test_datasets):
    """
    Verifies that the experiment runs correctly with a lightweight configuration.
    """
    # Instantiate concrete dependencies
    dataset_loader = MedMNISTDatasetLoader()
    variability_metric = ResNetMSDVariabilityMetric()
    similarity_metric = FIDSimilarityMetric()
    model = DiffusionModelWrapper()
    
    # Create experiment
    experiment = Experiment(
        datasets=test_datasets["datasets"],
        max_samples=test_datasets["max_samples"],
        image_size=test_datasets["image_size"],
        dataset_loader=dataset_loader,
        variability_metric=variability_metric,
        similarity_metric=similarity_metric,
        model=model
    )
    
    # Run experiment
    try:
        experiment.run()
    except Exception as e:
        pytest.fail(f"Experiment failed with error: {e}")

