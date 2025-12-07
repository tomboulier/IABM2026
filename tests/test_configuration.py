"""Tests for configuration module."""
import os
import tempfile
from pathlib import Path

import pytest

from src.infrastructure.configuration import (
    BatchTrainingConfiguration,
    TrainingConfiguration,
    _generate_output_path,
)


class TestGenerateOutputPath:
    """Tests for _generate_output_path function."""

    def test_with_max_samples_none(self):
        """Test filename generation when max_samples is None."""
        result = _generate_output_path(
            dataset="ChestMNIST",
            epochs=10,
            max_samples=None,
            image_size=28,
            batch_size=8,
            output_dir="models",
        )
        assert result == "models/chest_epoch-10_all-images_image-size-28_batch-8.weights.h5"

    def test_with_numeric_max_samples(self):
        """Test filename generation when max_samples is a number."""
        result = _generate_output_path(
            dataset="PathMNIST",
            epochs=50,
            max_samples=1000,
            image_size=64,
            batch_size=32,
            output_dir="models",
        )
        assert result == "models/path_epoch-50_1000-images_image-size-64_batch-32.weights.h5"

    def test_dataset_name_normalization(self):
        """Test that dataset names are normalized correctly."""
        # Test ChestMNIST -> chest
        result = _generate_output_path(
            dataset="ChestMNIST",
            epochs=10,
            max_samples=100,
            image_size=28,
            batch_size=8,
            output_dir="models",
        )
        assert "chest_" in result
        assert "mnist" not in result.lower()

        # Test BloodMNIST -> blood
        result = _generate_output_path(
            dataset="BloodMNIST",
            epochs=10,
            max_samples=100,
            image_size=28,
            batch_size=8,
            output_dir="models",
        )
        assert "blood_" in result
        assert "mnist" not in result.lower()

    def test_custom_output_dir(self):
        """Test that custom output directory is respected."""
        result = _generate_output_path(
            dataset="DermaMNIST",
            epochs=20,
            max_samples=500,
            image_size=32,
            batch_size=16,
            output_dir="custom/path",
        )
        assert result.startswith("custom/path/")
        assert result == "custom/path/derma_epoch-20_500-images_image-size-32_batch-16.weights.h5"


class TestTrainingConfiguration:
    """Tests for TrainingConfiguration class."""

    def test_post_init_generates_output_path(self):
        """Test that __post_init__ generates output path when not provided."""
        config = TrainingConfiguration(
            dataset="ChestMNIST",
            output=None,
            epochs=10,
            max_samples=100,
            image_size=28,
            batch_size=8,
        )
        assert config.output is not None
        assert isinstance(config.output, str)
        assert "chest_" in config.output
        assert "epoch-10" in config.output
        assert "100-images" in config.output

    def test_post_init_preserves_explicit_output(self):
        """Test that __post_init__ preserves explicitly set output path."""
        explicit_path = "custom/model.weights.h5"
        config = TrainingConfiguration(
            dataset="PathMNIST",
            output=explicit_path,
            epochs=20,
        )
        assert config.output == explicit_path

    def test_post_init_with_all_images(self):
        """Test that __post_init__ generates correct path for all images."""
        config = TrainingConfiguration(
            dataset="BloodMNIST",
            output=None,
            max_samples=None,
            epochs=15,
            image_size=64,
            batch_size=32,
        )
        assert "all-images" in config.output

    def test_load_from_toml(self, tmp_path):
        """Test loading configuration from TOML file."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[training]
dataset = "PathMNIST"
epochs = 25
max_samples = 500
image_size = 32
batch_size = 16
learning_rate = 0.001
output = "models/custom.weights.h5"
""")
        
        config = TrainingConfiguration.load(str(config_file))
        assert config.dataset == "PathMNIST"
        assert config.epochs == 25
        assert config.max_samples == 500
        assert config.image_size == 32
        assert config.batch_size == 16
        assert config.learning_rate == 0.001
        assert config.output == "models/custom.weights.h5"

    def test_load_from_toml_with_auto_output(self, tmp_path):
        """Test that auto-generation works when loading from TOML without output."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[training]
dataset = "ChestMNIST"
epochs = 10
batch_size = 8
""")
        
        config = TrainingConfiguration.load(str(config_file))
        assert config.output is not None
        assert "chest_" in config.output
        assert "epoch-10" in config.output

    def test_load_missing_file(self):
        """Test that loading from missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrainingConfiguration.load("/nonexistent/path/config.toml")


class TestBatchTrainingConfiguration:
    """Tests for BatchTrainingConfiguration class."""

    def test_load_with_defaults_and_trainings(self, tmp_path):
        """Test loading batch config with defaults and training entries."""
        config_file = tmp_path / "batch_config.toml"
        config_file.write_text("""
[batch.defaults]
image_size = 28
epochs = 10
batch_size = 8

[[batch.trainings]]
dataset = "PathMNIST"

[[batch.trainings]]
dataset = "ChestMNIST"
""")
        
        batch_config = BatchTrainingConfiguration.load(str(config_file))
        assert len(batch_config.trainings) == 2
        
        # Check first training
        config1 = batch_config.trainings[0]
        assert config1.dataset == "PathMNIST"
        assert config1.image_size == 28
        assert config1.epochs == 10
        assert config1.batch_size == 8
        
        # Check second training
        config2 = batch_config.trainings[1]
        assert config2.dataset == "ChestMNIST"
        assert config2.image_size == 28
        assert config2.epochs == 10
        assert config2.batch_size == 8

    def test_load_with_override(self, tmp_path):
        """Test that specific training configs override defaults."""
        config_file = tmp_path / "batch_config.toml"
        config_file.write_text("""
[batch.defaults]
epochs = 10
batch_size = 8
image_size = 28

[[batch.trainings]]
dataset = "PathMNIST"
epochs = 20

[[batch.trainings]]
dataset = "ChestMNIST"
batch_size = 16
""")
        
        batch_config = BatchTrainingConfiguration.load(str(config_file))
        
        # First training overrides epochs
        config1 = batch_config.trainings[0]
        assert config1.epochs == 20  # overridden
        assert config1.batch_size == 8  # from defaults
        
        # Second training overrides batch_size
        config2 = batch_config.trainings[1]
        assert config2.epochs == 10  # from defaults
        assert config2.batch_size == 16  # overridden

    def test_load_without_defaults(self, tmp_path):
        """Test loading batch config without defaults section."""
        config_file = tmp_path / "batch_config.toml"
        config_file.write_text("""
[[batch.trainings]]
dataset = "PathMNIST"
epochs = 15
batch_size = 16
image_size = 32

[[batch.trainings]]
dataset = "BloodMNIST"
epochs = 20
batch_size = 8
image_size = 64
""")
        
        batch_config = BatchTrainingConfiguration.load(str(config_file))
        assert len(batch_config.trainings) == 2
        
        config1 = batch_config.trainings[0]
        assert config1.dataset == "PathMNIST"
        assert config1.epochs == 15
        
        config2 = batch_config.trainings[1]
        assert config2.dataset == "BloodMNIST"
        assert config2.epochs == 20

    def test_load_empty_trainings(self, tmp_path):
        """Test loading batch config with no training entries."""
        config_file = tmp_path / "batch_config.toml"
        config_file.write_text("""
[batch.defaults]
epochs = 10
""")
        
        batch_config = BatchTrainingConfiguration.load(str(config_file))
        assert len(batch_config.trainings) == 0

    def test_load_missing_file(self):
        """Test that loading from missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BatchTrainingConfiguration.load("/nonexistent/path/config.toml")

    def test_auto_generated_output_paths(self, tmp_path):
        """Test that output paths are auto-generated for each training."""
        config_file = tmp_path / "batch_config.toml"
        config_file.write_text("""
[batch.defaults]
image_size = 28
epochs = 10
batch_size = 8

[[batch.trainings]]
dataset = "PathMNIST"

[[batch.trainings]]
dataset = "ChestMNIST"
max_samples = 1000
""")
        
        batch_config = BatchTrainingConfiguration.load(str(config_file))
        
        # Each training should have auto-generated output path
        config1 = batch_config.trainings[0]
        assert config1.output is not None
        assert "path_" in config1.output
        assert "all-images" in config1.output
        
        config2 = batch_config.trainings[1]
        assert config2.output is not None
        assert "chest_" in config2.output
        assert "1000-images" in config2.output
