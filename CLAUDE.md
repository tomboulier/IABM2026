# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (uses uv package manager)
make install

# Run the main experiment
make run
# Or with custom config:
uv run python main.py -c path/to/config.toml

# Run tests
uv run pytest tests/
# Run a single test file:
uv run pytest tests/test_tensorflow_model.py
# Run a specific test:
uv run pytest tests/test_end_to_end.py::test_main_with_test_configuration

# Type checking
uv run mypy src/

# Docker
make docker-build
make docker-run
```

## Architecture

This project follows **Clean Architecture** with strict separation between domain logic (framework-agnostic) and infrastructure (framework-specific implementations).

### Layer Structure

```
src/
├── domain/                    # Pure business logic, no external dependencies
│   ├── entities/              # Protocols: Dataset, Tensor, Metrics
│   ├── interfaces/            # Abstract base classes: DatasetLoader, Model, VariabilityMetric, SimilarityMetric
│   └── use_cases/             # Experiment orchestrator
└── infrastructure/            # Concrete implementations
    ├── loaders.py             # MedMNISTDatasetLoader
    ├── metrics.py             # ResNetMSDVariabilityMetric, FIDSimilarityMetric
    ├── datasets.py            # MedMNISTDatasetAdapter
    └── tensorflow/            # TensorFlow-specific: diffusion model (DDPM), U-Net, dataset adapter
```

### Key Design Patterns

- **Dependency Injection**: `main.py` wires concrete implementations into the `Experiment` use case
- **Adapter Pattern**: `MedMNISTDatasetAdapter` and `TensorFlowDatasetAdapter` bridge domain protocols to framework APIs
- **Protocol-based abstractions**: Domain entities use Python `Protocol` for framework independence

### Data Flow

1. `main.py` loads config and instantiates concrete dependencies
2. `Experiment.run()` orchestrates: load dataset → compute variability (ResNet-18 MSD) → train diffusion model → generate images → compute similarity (FID)
3. PyTorch handles feature extraction/metrics, TensorFlow handles the diffusion model

### Configuration

Experiments are configured via TOML files (default: `configuration.toml`):
```toml
[experiment]
datasets = ["PathMNIST", "ChestMNIST", "DermaMNIST", "BloodMNIST"]
max_samples = 10000
image_size = 224
```