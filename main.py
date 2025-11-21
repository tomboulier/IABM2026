from src.infrastructure.logging import setup_logging
from src.infrastructure.configuration import ExperimentConfiguration
from src.domain.use_cases.experiment import run_experiment

def main():
    # 1. Setup Logging
    setup_logging()
    
    # 2. Load Configuration
    config = ExperimentConfiguration.load("configuration.toml")
    
    # 3. Run Experiment
    run_experiment(config)

if __name__ == "__main__":
    main()
