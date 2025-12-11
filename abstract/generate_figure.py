"""Generate comparison figure for IABM abstract: real vs generated images.

Uses the same loading/generation methods as the Marimo notebook to ensure
consistent visualization between real and generated images.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project imports
from src.infrastructure.loaders import MedMNISTDatasetLoader
from src.infrastructure.tensorflow.diffusion_model import TensorFlowDiffusionModel

# Datasets to compare (all using epoch-1 models, consistent with results table)
# Format: (dataset_name, model_path, variability, fid)
DATASETS = [
    ("OrganCMNIST", "models/organc_epoch-1_all-images_image-size-28_batch-8.weights.h5", 122.1, 138.9),
    ("BloodMNIST", "models/blood_epoch-1_all-images_image-size-28_batch-8.weights.h5", 96.8, 313.2),
    ("RetinaMNIST", "models/retina_epoch-1_all-images_image-size-28_batch-8.weights.h5", 85.0, 574.0),
]

PROJECT_ROOT = Path(__file__).parent.parent
ABSTRACT_DIR = Path(__file__).parent

# ImageNet normalization stats (same as in datasets.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

N_IMAGES = 5


def load_real_samples(dataset_name: str, n_samples: int = N_IMAGES):
    """Load real samples using the same loader as the notebook."""
    loader = MedMNISTDatasetLoader()
    dataset = loader.load(dataset_name, max_samples=None, image_size=28)

    # Get random sample indices
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    images = []
    for idx in indices:
        img = dataset[idx]
        if isinstance(img, tuple):
            img = img[0]
        # Convert to numpy
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.array(img)
        # Handle channel ordering (C, H, W) -> (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        # Denormalize from ImageNet stats
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img, 0, 1)
        images.append(img)

    return images


def load_generated_samples(model_path: str, n_samples: int = N_IMAGES):
    """Generate samples using the same method as the notebook."""
    model = TensorFlowDiffusionModel(
        image_size=28,
        num_channels=3,
    )
    full_path = PROJECT_ROOT / model_path
    model.load(str(full_path))

    # Generate images (already denormalized and clipped to [0, 1])
    generated = model.generate_images(n=n_samples)

    images = []
    for i in range(n_samples):
        img = generated[i]
        img = np.clip(img, 0, 1)
        images.append(img)

    return images


def create_comparison_figure():
    """Create a figure comparing real vs generated images."""
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(8, 6))

    for i, (dataset_name, model_path, variability, fid) in enumerate(DATASETS):
        print(f"Processing {dataset_name}...")

        # Load real samples
        real_images = load_real_samples(dataset_name, N_IMAGES)

        # Generate samples
        gen_images = load_generated_samples(model_path, N_IMAGES)

        # Create mosaic of real images
        real_mosaic = np.hstack(real_images)

        # Create mosaic of generated images
        gen_mosaic = np.hstack(gen_images)

        # Plot real images
        axes[i, 0].imshow(real_mosaic)
        axes[i, 0].set_ylabel(f"{dataset_name}\nVar={variability:.0f}, FID={fid:.0f}",
                              fontsize=9)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Plot generated images
        axes[i, 1].imshow(gen_mosaic)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    # Column titles
    axes[0, 0].set_title("Images réelles", fontsize=11, fontweight='bold')
    axes[0, 1].set_title("Images générées", fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = ABSTRACT_DIR / "figure_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {output_path}")

    # Also save as PDF for LaTeX
    pdf_path = ABSTRACT_DIR / "figure_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    create_comparison_figure()
