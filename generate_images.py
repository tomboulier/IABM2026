"""
CLI entry point for generating images from a pre-trained diffusion model.

Usage:
    python generate_images.py -w models/pathmnist.weights.h5 -o output/
    python generate_images.py -w models/blood.weights.h5 -o output/ -n 10 --image-size 64
"""
import argparse

# Suppress TensorFlow logging before importing TF
from src.infrastructure.tensorflow.observability import suppress_tensorflow_logging

suppress_tensorflow_logging()

from src.domain.use_cases.generate_and_save_images import GenerateAndSaveImages
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
        description="Generate images from a pre-trained diffusion model."
    )
    parser.add_argument(
        "-w",
        "--weights",
        required=True,
        help="Path to the saved model weights (.weights.h5 file)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./outputs",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=16,
        help="Number of images to generate (default: 16)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Image size matching the trained model (default: 64)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=20,
        help="Number of diffusion steps for generation (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """
    Main entry point for generating images.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. If None, uses sys.argv.
    """
    setup_logging()
    args = parse_args(argv)

    # Instantiate the model (weights will be loaded by use-case)
    model = TensorFlowDiffusionModel(
        image_size=args.image_size,
        num_channels=3,  # MedMNIST datasets are RGB
        plot_diffusion_steps=args.diffusion_steps,
    )

    # Create and run use-case
    use_case = GenerateAndSaveImages(
        model=model,
        num_images=args.num_images,
        output_dir=args.output,
        weights_path=args.weights,
    )
    use_case.run()

    print(f"Generated {args.num_images} images in {args.output}/")


if __name__ == "__main__":
    main()
