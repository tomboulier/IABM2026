"""
CLI entry point for training and saving a diffusion model.

Usage:
    python train_model.py --dataset PathMNIST --output models/pathmnist.weights.h5
    python train_model.py -d BloodMNIST -o models/blood.weights.h5 --epochs 10
"""
import argparse
import os

from src.domain.use_cases.train_and_save_model import TrainAndSaveModel
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
        "-d",
        "--dataset",
        required=True,
        help="Name of the MedMNIST dataset (e.g., PathMNIST, BloodMNIST)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
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

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Instantiate dependencies
    dataset_loader = MedMNISTDatasetLoader()
    model = TensorFlowDiffusionModel(
        image_size=args.image_size,
        num_channels=3,  # MedMNIST datasets are converted to RGB
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Create and run use-case
    use_case = TrainAndSaveModel(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        image_size=args.image_size,
        dataset_loader=dataset_loader,
        model=model,
        output_path=args.output,
    )
    use_case.run()


if __name__ == "__main__":
    main()
