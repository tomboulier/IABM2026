"""
CLI entry point for training and saving a diffusion model.

Usage with config file:
    python train_model.py -c config/train_pathmnist.toml

Usage with command-line args:
    python train_model.py -d PathMNIST -o models/pathmnist.weights.h5
    python train_model.py -d BloodMNIST -o models/blood.weights.h5 --epochs 10
"""
import argparse
import os

# Suppress TensorFlow logging before importing TF
from src.infrastructure.tensorflow.observability import (
    ImageSavingTracker,
    suppress_tensorflow_logging,
)

suppress_tensorflow_logging()

from src.domain.use_cases.train_and_save_model import TrainAndSaveModel
from src.infrastructure.configuration import TrainingConfiguration
from src.infrastructure.loaders import MedMNISTDatasetLoader
from src.infrastructure.logging import setup_logging
from src.infrastructure.tensorflow.diffusion_model import TensorFlowDiffusionModel


def parse_args(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on a MedMNIST dataset and save weights."
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to TOML configuration file (if provided, other args are ignored)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Name of the MedMNIST dataset (e.g., PathMNIST, BloodMNIST)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for saved model weights (e.g., models/model.weights.h5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (default: all)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Image size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--samples-output-dir",
        type=str,
        default="./outputs",
        help="Directory for sample images during training (default: ./output)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """
    Main entry point for training and saving a model.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. If None, uses sys.argv.
    """
    setup_logging()
    args = parse_args(argv)

    # Load configuration from file or command-line args
    if args.config:
        config = TrainingConfiguration.load(args.config)
    else:
        # Validate required args when not using config file
        if not args.dataset or not args.output:
            raise ValueError(
                "Either --config or both --dataset and --output are required"
            )
        config = TrainingConfiguration(
            dataset=args.dataset,
            output=args.output,
            max_samples=args.max_samples,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            samples_output_dir=args.samples_output_dir,
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(config.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Instantiate dependencies
    dataset_loader = MedMNISTDatasetLoader()
    tracker = ImageSavingTracker(output_dir=config.samples_output_dir)
    model = TensorFlowDiffusionModel(
        image_size=config.image_size,
        num_channels=3,  # MedMNIST datasets are converted to RGB
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        tracker=tracker,
    )

    # Create and run use-case
    use_case = TrainAndSaveModel(
        dataset_name=config.dataset,
        max_samples=config.max_samples,
        image_size=config.image_size,
        dataset_loader=dataset_loader,
        model=model,
        output_path=config.output,
    )
    use_case.run()


if __name__ == "__main__":
    main()
