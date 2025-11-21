import pytest
import tomli_w
from src.infrastructure.configuration import ExperimentConfiguration
from src.domain.use_cases.experiment import run_experiment

@pytest.fixture
def test_config_path(tmp_path):
    """
    Creates a temporary configuration file for testing.
    """
    config_data = {
        "experiment": {
            "datasets": ["ChestMNIST"],
            "max_samples": 100,
            "image_size": 28
        }
    }
    config_file = tmp_path / "test_config.toml"
    with open(config_file, "wb") as f:
        tomli_w.dump(config_data, f)
    return str(config_file)

def test_run_experiment(test_config_path):
    """
    Verifies that the experiment runs correctly with a lightweight configuration.
    """
    # Load configuration
    config = ExperimentConfiguration.load(test_config_path)
    
    # Run experiment
    try:
        run_experiment(config)
    except Exception as e:
        pytest.fail(f"Experiment failed with error: {e}")
