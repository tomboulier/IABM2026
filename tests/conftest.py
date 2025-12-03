"""Pytest configuration and shared fixtures."""
import pytest
from fixtures.dummy_model import DummyModel


@pytest.fixture
def dummy_model():
    """
    Provide a DummyModel instance for tests.
    
    Returns:
        DummyModel: A new DummyModel instance.
    """
    return DummyModel()


@pytest.fixture
def test_datasets():
    """
    Provide a lightweight dataset configuration for tests.
    
    Returns:
        dict: Configuration with the following keys:
            datasets (list[str]): Dataset names to use (e.g., ["ChestMNIST"]).
            max_samples (int): Maximum number of samples to use per dataset.
            image_size (int): Target image size in pixels (square).
    """
    return {
        "datasets": ["ChestMNIST"],
        "max_samples": 100,
        "image_size": 28
    }