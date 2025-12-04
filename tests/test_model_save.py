"""Tests for model save functionality."""
import os
import tempfile

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


class TestTensorFlowDiffusionModelSave:
    """Tests for TensorFlowDiffusionModel save functionality."""

    def test_save_creates_weights_file(self):
        """
        Calling save(path) should create a weights file at the specified path.
        """
        from src.infrastructure.tensorflow.diffusion_model import (
            TensorFlowDiffusionModel,
        )

        model = TensorFlowDiffusionModel(
            image_size=28,
            num_channels=1,
            epochs=0,  # No training needed for this test
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "model.weights.h5")
            model.save(weights_path)

            assert os.path.exists(weights_path), (
                f"Weights file should exist at {weights_path}"
            )

    def test_saved_model_can_be_loaded(self):
        """
        A saved model should be loadable by a new model instance.
        """
        from src.infrastructure.tensorflow.diffusion_model import (
            TensorFlowDiffusionModel,
        )

        model = TensorFlowDiffusionModel(
            image_size=28,
            num_channels=1,
            epochs=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "model.weights.h5")
            model.save(weights_path)

            # Load into a new model instance
            loaded_model = TensorFlowDiffusionModel(
                image_size=28,
                num_channels=1,
                epochs=0,
                load_weights_path=weights_path,
            )

            # Verify weights were loaded (no exception raised)
            assert loaded_model is not None
