"""Pytest configuration and shared fixtures."""
import pytest
from fixtures.dummy_model import DummyModel


@pytest.fixture
def dummy_model():
    """Provides a DummyModel instance for fast testing."""
    return DummyModel()


@pytest.fixture
def test_datasets():
    """Lightweight test configuration."""
    return {
        "datasets": ["ChestMNIST"],
        "max_samples": 100,
        "image_size": 28
    }
