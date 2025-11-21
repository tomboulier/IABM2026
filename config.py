import tomllib
import os

def load_config(config_path: str = "config.toml") -> dict:
    """
    Loads the configuration from a TOML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        A dictionary containing the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "rb") as f:
        return tomllib.load(f)
