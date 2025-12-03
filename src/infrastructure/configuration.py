import tomllib
import os
from dataclasses import dataclass

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