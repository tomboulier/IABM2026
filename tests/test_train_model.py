"""Tests for train_model.py CLI."""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from train_model import main, parse_args, train_single
from src.infrastructure.configuration import TrainingConfiguration


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_config_arg(self):
        """Test parsing --config argument."""
        args = parse_args(["-c", "config.toml"])
        assert args.config == "config.toml"

    def test_parse_batch_flag(self):
        """Test parsing --batch flag."""
        args = parse_args(["--batch", "-c", "config.toml"])
        assert args.batch is True

    def test_parse_dataset_and_output(self):
        """Test parsing dataset and output arguments."""
        args = parse_args(["-d", "PathMNIST", "-o", "model.h5"])
        assert args.dataset == "PathMNIST"
        assert args.output == "model.h5"

    def test_parse_training_params(self):
        """Test parsing training parameter arguments."""
        args = parse_args([
            "-d", "ChestMNIST",
            "--epochs", "20",
            "--batch-size", "16",
            "--learning-rate", "0.0001",
            "--image-size", "32",
        ])
        assert args.dataset == "ChestMNIST"
        assert args.epochs == 20
        assert args.batch_size == 16
        assert args.learning_rate == 0.0001
        assert args.image_size == 32


class TestBatchTrainingMode:
    """Tests for batch training mode."""

    @patch("train_model.train_single")
    @patch("train_model.BatchTrainingConfiguration.load")
    def test_batch_training_loads_config(self, mock_load, mock_train_single, tmp_path):
        """Test that batch training loads the batch configuration."""
        # Create mock batch config
        mock_config1 = TrainingConfiguration(
            dataset="PathMNIST",
            epochs=10,
            batch_size=8,
        )
        mock_config2 = TrainingConfiguration(
            dataset="ChestMNIST",
            epochs=10,
            batch_size=8,
        )
        mock_batch = MagicMock()
        mock_batch.trainings = [mock_config1, mock_config2]
        mock_load.return_value = mock_batch

        # Run batch training
        config_path = str(tmp_path / "batch.toml")
        main(["-c", config_path, "--batch"])

        # Verify config was loaded
        mock_load.assert_called_once_with(config_path)

    @patch("train_model.train_single")
    @patch("train_model.BatchTrainingConfiguration.load")
    def test_batch_training_executes_all_trainings(self, mock_load, mock_train_single, tmp_path):
        """Test that batch training executes all training configurations."""
        # Create mock batch config with 3 trainings
        configs = [
            TrainingConfiguration(dataset="PathMNIST", epochs=10),
            TrainingConfiguration(dataset="ChestMNIST", epochs=15),
            TrainingConfiguration(dataset="BloodMNIST", epochs=20),
        ]
        mock_batch = MagicMock()
        mock_batch.trainings = configs
        mock_load.return_value = mock_batch

        # Run batch training
        config_path = str(tmp_path / "batch.toml")
        main(["-c", config_path, "--batch"])

        # Verify train_single was called for each config
        assert mock_train_single.call_count == 3
        for i, config in enumerate(configs):
            assert mock_train_single.call_args_list[i][0][0] == config

    @patch("train_model.BatchTrainingConfiguration.load")
    def test_batch_training_requires_config(self, mock_load):
        """Test that --batch requires --config to be specified."""
        with pytest.raises(ValueError, match="--batch requires --config"):
            main(["--batch"])

    @patch("train_model.train_single")
    @patch("train_model.BatchTrainingConfiguration.load")
    def test_batch_training_with_empty_config(self, mock_load, mock_train_single, tmp_path):
        """Test batch training with empty configuration (no trainings)."""
        # Create mock batch config with no trainings
        mock_batch = MagicMock()
        mock_batch.trainings = []
        mock_load.return_value = mock_batch

        # Run batch training with empty config
        config_path = str(tmp_path / "batch.toml")
        
        # Should raise ValueError for empty batch configuration
        with pytest.raises(ValueError, match="Empty batch configuration"):
            main(["-c", config_path, "--batch"])

        # train_single should not be called
        mock_train_single.assert_not_called()


class TestSingleTrainingMode:
    """Tests for single training mode."""

    @patch("train_model.train_single")
    @patch("train_model.TrainingConfiguration.load")
    def test_single_training_with_config_file(self, mock_load, mock_train_single, tmp_path):
        """Test single training mode with config file."""
        mock_config = TrainingConfiguration(
            dataset="PathMNIST",
            epochs=10,
        )
        mock_load.return_value = mock_config

        config_path = str(tmp_path / "config.toml")
        main(["-c", config_path])

        mock_load.assert_called_once_with(config_path)
        mock_train_single.assert_called_once_with(mock_config)

    @patch("train_model.train_single")
    def test_single_training_with_cli_args(self, mock_train_single):
        """Test single training mode with command-line arguments."""
        main(["-d", "PathMNIST", "-o", "model.h5"])

        # Verify train_single was called with correct config
        mock_train_single.assert_called_once()
        config = mock_train_single.call_args[0][0]
        assert config.dataset == "PathMNIST"
        assert config.output == "model.h5"

    def test_single_training_requires_dataset_or_config(self):
        """Test that either --config or --dataset is required."""
        with pytest.raises(ValueError, match="Either --config or --dataset is required"):
            main([])


class TestTrainSingle:
    """Tests for train_single function."""

    @patch("train_model.TrainAndSaveModel")
    @patch("train_model.TensorFlowDiffusionModel")
    @patch("train_model.ImageSavingTracker")
    @patch("train_model.MedMNISTDatasetLoader")
    @patch("os.makedirs")
    def test_train_single_creates_output_dir(
        self,
        mock_makedirs,
        mock_loader,
        mock_tracker,
        mock_model,
        mock_use_case,
        tmp_path,
    ):
        """Test that train_single creates output directory."""
        config = TrainingConfiguration(
            dataset="PathMNIST",
            output=str(tmp_path / "models" / "test.h5"),
            epochs=10,
        )
        
        train_single(config)
        
        # Verify makedirs was called with the output directory
        mock_makedirs.assert_called_once_with(str(tmp_path / "models"), exist_ok=True)

    @patch("train_model.TrainAndSaveModel")
    @patch("train_model.TensorFlowDiffusionModel")
    @patch("train_model.ImageSavingTracker")
    @patch("train_model.MedMNISTDatasetLoader")
    def test_train_single_instantiates_dependencies(
        self,
        mock_loader_class,
        mock_tracker_class,
        mock_model_class,
        mock_use_case_class,
    ):
        """Test that train_single instantiates all required dependencies."""
        config = TrainingConfiguration(
            dataset="ChestMNIST",
            output="test.h5",
            epochs=15,
            batch_size=16,
            image_size=32,
            learning_rate=0.0001,
        )
        
        train_single(config)
        
        # Verify all components were instantiated
        mock_loader_class.assert_called_once()
        mock_tracker_class.assert_called_once_with(output_dir=config.samples_output_dir)
        mock_model_class.assert_called_once()
        
        # Verify model was instantiated with correct params
        model_call_kwargs = mock_model_class.call_args[1]
        assert model_call_kwargs["image_size"] == 32
        assert model_call_kwargs["epochs"] == 15
        assert model_call_kwargs["batch_size"] == 16
        assert model_call_kwargs["learning_rate"] == 0.0001

    @patch("train_model.TrainAndSaveModel")
    @patch("train_model.TensorFlowDiffusionModel")
    @patch("train_model.ImageSavingTracker")
    @patch("train_model.MedMNISTDatasetLoader")
    def test_train_single_runs_use_case(
        self,
        mock_loader_class,
        mock_tracker_class,
        mock_model_class,
        mock_use_case_class,
    ):
        """Test that train_single creates and runs the use case."""
        config = TrainingConfiguration(
            dataset="BloodMNIST",
            output="test.h5",
            max_samples=1000,
        )
        
        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case
        
        train_single(config)
        
        # Verify use case was created with correct params
        mock_use_case_class.assert_called_once()
        call_kwargs = mock_use_case_class.call_args[1]
        assert call_kwargs["dataset_name"] == "BloodMNIST"
        assert call_kwargs["max_samples"] == 1000
        assert call_kwargs["output_path"] == "test.h5"
        
        # Verify run was called
        mock_use_case.run.assert_called_once()
