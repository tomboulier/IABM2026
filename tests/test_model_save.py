"""Tests for model save functionality."""
import pytest
from abc import ABC

from src.domain.interfaces.model import Model


class TestModelInterface:
    """Tests for the Model abstract interface."""

    def test_model_has_save_method(self):
        """
        The Model interface should define an abstract save method.

        This ensures all concrete model implementations must provide
        a way to persist the trained model to disk.
        """
        assert hasattr(Model, "save"), "Model interface should have a save method"

    def test_save_is_abstract(self):
        """
        The save method should be abstract, requiring implementation
        by concrete subclasses.
        """
        # Check that save is in the abstract methods
        assert "save" in Model.__abstractmethods__, (
            "save should be an abstract method"
        )
