import os
import tomllib
from dataclasses import dataclass


@dataclass
class TrainingConfiguration:
    """Configuration for model training."""

    dataset: str
    output: str
    max_samples: int | None = None
    image_size: int = 64
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    samples_output_dir: str = "./output"

    @classmethod
    def load(cls, config_path: str) -> "TrainingConfiguration":
        """
        Load training configuration from a TOML file.

        Parameters
        ----------
        config_path : str
            Filesystem path to a TOML file containing a "training" table.

        Returns
        -------
        TrainingConfiguration
            Instance populated from the "training" table.

        Raises
        ------
        FileNotFoundError
            If no file exists at `config_path`.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        training_data = data.get("training", {})
        return cls(**training_data)


@dataclass
class GenerationConfiguration:
    """Configuration for image generation."""

    weights: str
    output: str
    num_images: int = 16
    image_size: int = 64
    diffusion_steps: int = 20

    @classmethod
    def load(cls, config_path: str) -> "GenerationConfiguration":
        """
        Load generation configuration from a TOML file.

        Parameters
        ----------
        config_path : str
            Filesystem path to a TOML file containing a "generation" table.

        Returns
        -------
        GenerationConfiguration
            Instance populated from the "generation" table.

        Raises
        ------
        FileNotFoundError
            If no file exists at `config_path`.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        generation_data = data.get("generation", {})
        return cls(**generation_data)


@dataclass
class ExperimentConfiguration:
    datasets: list[str]
    max_samples: int | None = None
    image_size: int = 224
    variability_metric: str = "resnet_msd"
    similarity_metric: str = "fid"
    model_type: str = "diffusion"

    @classmethod
    def load(cls, config_path: str) -> "ExperimentConfiguration":
        """
        Load experiment configuration from a TOML file.
        
        Parameters:
            config_path (str): Filesystem path to a TOML file containing an "experiment" table.
        
        Returns:
            ExperimentConfiguration: Instance populated from the "experiment" table; fields not present use their dataclass defaults.
        
        Raises:
            FileNotFoundError: If no file exists at `config_path`.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
            
        experiment_data = data.get("experiment", {})
        return cls(**experiment_data)
