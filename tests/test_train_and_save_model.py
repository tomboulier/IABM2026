"""Tests for TrainAndSaveModel use-case."""
import os
import tempfile
from unittest.mock import MagicMock, call

import pytest

from src.domain.interfaces.dataset_loader import DatasetLoader
from src.domain.interfaces.model import Model


class TestTrainAndSaveModel:
    """Tests for the TrainAndSaveModel use-case."""

    def test_use_case_exists(self):
        """
        The TrainAndSaveModel use-case class should exist in the domain layer.
        """
        from src.domain.use_cases.train_and_save_model import TrainAndSaveModel

        assert TrainAndSaveModel is not None

    def test_run_loads_dataset(self):
        """
        Running the use-case should load the specified dataset.
        """
        from src.domain.use_cases.train_and_save_model import TrainAndSaveModel

        # Create mocks
        mock_loader = MagicMock(spec=DatasetLoader)
        mock_dataset = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_model = MagicMock(spec=Model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.weights.h5")

            use_case = TrainAndSaveModel(
                dataset_name="PathMNIST",
                max_samples=100,
                image_size=28,
                dataset_loader=mock_loader,
                model=mock_model,
                output_path=output_path,
            )
            use_case.run()

            mock_loader.load.assert_called_once_with("PathMNIST", 100, 28)

    def test_run_trains_model_with_dataset(self):
        """
        Running the use-case should train the model on the loaded dataset.
        """
        from src.domain.use_cases.train_and_save_model import TrainAndSaveModel

        mock_loader = MagicMock(spec=DatasetLoader)
        mock_dataset = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_model = MagicMock(spec=Model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.weights.h5")

            use_case = TrainAndSaveModel(
                dataset_name="PathMNIST",
                max_samples=100,
                image_size=28,
                dataset_loader=mock_loader,
                model=mock_model,
                output_path=output_path,
            )
            use_case.run()

            mock_model.train.assert_called_once_with(
                mock_dataset, dataset_name="PathMNIST"
            )

    def test_run_saves_model_after_training(self):
        """
        Running the use-case should save the model to the specified path
        after training completes.
        """
        from src.domain.use_cases.train_and_save_model import TrainAndSaveModel

        mock_loader = MagicMock(spec=DatasetLoader)
        mock_dataset = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_model = MagicMock(spec=Model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.weights.h5")

            use_case = TrainAndSaveModel(
                dataset_name="PathMNIST",
                max_samples=100,
                image_size=28,
                dataset_loader=mock_loader,
                model=mock_model,
                output_path=output_path,
            )
            use_case.run()

            mock_model.save.assert_called_once_with(output_path)

    def test_operations_happen_in_correct_order(self):
        """
        The use-case should execute operations in order:
        1. Load dataset
        2. Train model
        3. Save model
        """
        from src.domain.use_cases.train_and_save_model import TrainAndSaveModel

        call_order = []

        mock_loader = MagicMock(spec=DatasetLoader)
        mock_dataset = MagicMock()
        mock_loader.load.side_effect = lambda *args: (
            call_order.append("load"),
            mock_dataset,
        )[1]

        mock_model = MagicMock(spec=Model)
        mock_model.train.side_effect = lambda *args, **kwargs: call_order.append("train")
        mock_model.save.side_effect = lambda *args, **kwargs: call_order.append("save")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.weights.h5")

            use_case = TrainAndSaveModel(
                dataset_name="PathMNIST",
                max_samples=100,
                image_size=28,
                dataset_loader=mock_loader,
                model=mock_model,
                output_path=output_path,
            )
            use_case.run()

            assert call_order == ["load", "train", "save"], (
                f"Operations should happen in order: load, train, save. "
                f"Got: {call_order}"
            )

    def test_run_raises_if_output_path_invalid(self):
        """
        Running the use-case should raise ValueError if output path
        does not end with '.weights.h5'.
        """
        from src.domain.use_cases.train_and_save_model import TrainAndSaveModel

        mock_loader = MagicMock(spec=DatasetLoader)
        mock_model = MagicMock(spec=Model)

        use_case = TrainAndSaveModel(
            dataset_name="PathMNIST",
            max_samples=100,
            image_size=28,
            dataset_loader=mock_loader,
            model=mock_model,
            output_path="/tmp/bad_path.weights.h",  # Missing '5'
        )

        with pytest.raises(ValueError, match="must end with '.weights.h5'"):
            use_case.run()

        # Ensure no training happened
        mock_loader.load.assert_not_called()
        mock_model.train.assert_not_called()
