import os
import tomllib
from dataclasses import dataclass


def _generate_output_path(
    dataset: str,
    epochs: int,
    max_samples: int | None,
    image_size: int,
    batch_size: int,
    output_dir: str = "models",
) -> str:
    """
    Generate a descriptive output path for model weights.

    Parameters
    ----------
    dataset : str
        Name of the dataset (e.g., "ChestMNIST").
    epochs : int
        Number of training epochs.
    max_samples : int | None
        Maximum samples used, or None for all.
    image_size : int
        Image size in pixels.
    batch_size : int
        Training batch size.
    output_dir : str
        Directory for output files.

    Returns
    -------
    str
        Generated path like "models/chest_epoch-10_all-images_image-size-28_batch-8.weights.h5"
    """
    dataset_lower = dataset.lower().replace("mnist", "")
    samples_str = "all-images" if max_samples is None else f"{max_samples}-images"
    filename = (
        f"{dataset_lower}_epoch-{epochs}_{samples_str}_"
        f"image-size-{image_size}_batch-{batch_size}.weights.h5"
    )
    return os.path.join(output_dir, filename)


@dataclass
class TrainingConfiguration:
    """Configuration for model training."""

    dataset: str
    output: str | None = None  # Note: After __post_init__, this is always a str
    max_samples: int | None = None
    image_size: int = 64
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    samples_output_dir: str = "./output"
    output_dir: str = "models"  # Directory for model weights storage

    def __post_init__(self):
        """Generate output path if not provided.
        
        After this method executes, self.output is guaranteed to be a str.
        """
        if self.output is None:
            self.output = _generate_output_path(
                dataset=self.dataset,
                epochs=self.epochs,
                max_samples=self.max_samples,
                image_size=self.image_size,
                batch_size=self.batch_size,
                output_dir=self.output_dir,
            )

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
class BatchTrainingConfiguration:
    """Configuration for batch training multiple models."""

    trainings: list[TrainingConfiguration]

    @classmethod
    def load(cls, config_path: str) -> "BatchTrainingConfiguration":
        """
        Load batch training configuration from a TOML file.

        The file should contain a [batch] section with [[batch.trainings]] entries.
        Global defaults can be set in [batch.defaults].

        Parameters
        ----------
        config_path : str
            Filesystem path to a TOML file.

        Returns
        -------
        BatchTrainingConfiguration
            Instance with list of TrainingConfiguration objects.

        Raises
        ------
        FileNotFoundError
            If no file exists at `config_path`.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        batch_data = data.get("batch", {})
        defaults = batch_data.get("defaults", {})
        trainings_data = batch_data.get("trainings", [])

        trainings = []
        for training_data in trainings_data:
            # Merge defaults with specific training config
            merged = {**defaults, **training_data}
            trainings.append(TrainingConfiguration(**merged))

        return cls(trainings=trainings)


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
