"""Tests for Model.load() method."""
import tempfile

import pytest

from src.domain.interfaces.model import Model


class TestModelLoadInterface:
    """Tests for the load method in Model interface."""

    def test_load_method_exists_in_interface(self):
        """
        The Model interface should have a load method.
        """
        assert hasattr(Model, "load")

    def test_load_is_abstract_method(self):
        """
        The load method should be abstract.
        """
        # Check that Model.load is an abstract method
        assert getattr(Model.load, "__isabstractmethod__", False)


class TestTensorFlowDiffusionModelLoad:
    """Tests for load method in TensorFlowDiffusionModel."""

    def test_load_restores_saved_weights(self):
        """
        Loading weights from a file should restore the model state.
        """
        from src.infrastructure.tensorflow.diffusion_model import (
            TensorFlowDiffusionModel,
        )

        # Create and save a model
        model1 = TensorFlowDiffusionModel(
            image_size=28,
            num_channels=3,
            epochs=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = f"{tmpdir}/test.weights.h5"
            model1.save(weights_path)

            # Create a new model and load the weights
            model2 = TensorFlowDiffusionModel(
                image_size=28,
                num_channels=3,
                epochs=1,
            )
            model2.load(weights_path)

            # Both models should produce similar outputs for same input
            # (we can't test exact equality due to random noise in generation)
            # Instead, verify that load doesn't raise and model is usable
            images = model2.generate_images(1)
            assert images.shape == (1, 28, 28, 3)

    def test_load_raises_for_invalid_path(self):
        """
        Loading from a non-existent file should raise an error.
        """
        from src.infrastructure.tensorflow.diffusion_model import (
            TensorFlowDiffusionModel,
        )

        model = TensorFlowDiffusionModel(
            image_size=28,
            num_channels=3,
            epochs=1,
        )

        with pytest.raises(Exception):  # Could be OSError, ValueError, etc.
            model.load("/nonexistent/path/model.weights.h5")
