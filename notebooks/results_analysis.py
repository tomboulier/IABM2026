import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MedMNIST Experiment Results Analysis

    This notebook analyzes the results of the variability and similarity experiments
    on MedMNIST datasets using pre-trained diffusion models.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    return pd, plt


@app.cell
def _(pd):
    from pathlib import Path

    # Load experiment results (use path relative to this notebook file)
    notebook_dir = Path(__file__).parent
    results_path = notebook_dir / "../results/experiment_results.csv"
    df = pd.read_csv(results_path)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variability vs Similarity

    Scatter plot showing the relationship between dataset variability
    (Mean Squared Distance to Centroid) and similarity score (FID).

    - **Variability**: Higher values indicate more diverse datasets
    - **Similarity (FID)**: Lower values indicate generated images closer to real ones
    """)
    return


@app.cell
def _(df, plt):
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

    # Create scatter plot
    ax_scatter.scatter(
        df["variability"],
        1/df["similarity"],
        s=100,
        alpha=0.7,
        c=range(len(df)),
        cmap="viridis",
    )

    # Add labels for each point
    for _, row_scatter in df.iterrows():
        ax_scatter.annotate(
            row_scatter["dataset"].replace("MNIST", ""),
            (row_scatter["variability"], 1/row_scatter["similarity"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax_scatter.set_xlabel("Variability (MSD to Centroid)", fontsize=12)
    ax_scatter.set_ylabel("Similarity (1/FID)", fontsize=12)
    ax_scatter.set_title("Dataset Variability vs Model Similarity", fontsize=14)
    ax_scatter.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_scatter
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Testing

    Select a dataset and number of images to compare real vs generated samples.
    """)
    return


@app.cell
def _(df, mo):
    # Create dropdown for dataset selection
    dataset_dropdown = mo.ui.dropdown(
        options=df["dataset"].tolist(),
        value=df["dataset"].iloc[0],
        label="Dataset",
    )
    # Create slider for number of images
    num_images_slider = mo.ui.slider(
        start=1,
        stop=10,
        value=5,
        label="Number of images",
    )
    mo.hstack([dataset_dropdown, num_images_slider])
    return dataset_dropdown, num_images_slider


@app.cell
def _(dataset_dropdown, df):
    # Get weights path for selected dataset
    selected_dataset = dataset_dropdown.value
    weights_path_row = df[df["dataset"] == selected_dataset]["model_weights"]
    weights_path = weights_path_row.iloc[0] if len(weights_path_row) > 0 else None
    return selected_dataset, weights_path


@app.cell
def _():
    # Import required modules for model and data loading
    import numpy as np
    from pathlib import Path as PathLib

    # Get project root
    project_root = PathLib(__file__).parent.parent
    return np, project_root


@app.cell
def _(np, selected_dataset):
    # Load dataset
    from src.infrastructure.loaders import MedMNISTDatasetLoader

    loader = MedMNISTDatasetLoader()
    dataset = loader.load(selected_dataset, max_samples=None, image_size=28)

    # Get random sample indices
    sample_indices = np.random.choice(len(dataset), size=min(10, len(dataset)), replace=False)
    return dataset, sample_indices


@app.cell
def _(dataset, np, num_images_slider, plt, sample_indices):
    # Display random real images from dataset
    n_real = num_images_slider.value
    indices_to_show = sample_indices[:n_real]

    # ImageNet normalization stats used in dataset loading
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    fig_real, axes_real = plt.subplots(1, n_real, figsize=(n_real * 2, 2))
    if n_real == 1:
        axes_real = [axes_real]

    for k, sample_idx in enumerate(indices_to_show):
        real_img = dataset[sample_idx]
        if isinstance(real_img, tuple):
            real_img = real_img[0]
        # Convert to numpy if needed
        if hasattr(real_img, "numpy"):
            real_img = real_img.numpy()
        real_img = np.array(real_img)
        # Handle channel ordering (C, H, W) -> (H, W, C)
        if real_img.ndim == 3 and real_img.shape[0] in [1, 3]:
            real_img = np.transpose(real_img, (1, 2, 0))
        # Denormalize from ImageNet stats
        real_img = real_img * imagenet_std + imagenet_mean
        real_img = np.clip(real_img, 0, 1)
        # Handle grayscale
        if real_img.ndim == 3 and real_img.shape[-1] == 1:
            real_img = real_img.squeeze(-1)
            axes_real[k].imshow(real_img, cmap="gray")
        else:
            axes_real[k].imshow(real_img)
        axes_real[k].axis("off")
        axes_real[k].set_title(f"Real #{sample_idx}")

    fig_real.suptitle("Real Images from Dataset", fontsize=12)
    plt.tight_layout()
    fig_real
    return


@app.cell
def _(num_images_slider, project_root, weights_path):
    # Load model and generate images
    from src.infrastructure.tensorflow.diffusion_model import TensorFlowDiffusionModel

    model = TensorFlowDiffusionModel(
        image_size=28,
        num_channels=3,
    )
    full_weights_path = project_root / weights_path
    model.load(str(full_weights_path))

    # Generate images
    n_gen = num_images_slider.value
    generated_images = model.generate_images(n=n_gen)
    return generated_images, n_gen


@app.cell
def _(generated_images, n_gen, np, plt):
    # Display generated images
    fig_gen, axes_gen = plt.subplots(1, n_gen, figsize=(n_gen * 2, 2))
    if n_gen == 1:
        axes_gen = [axes_gen]

    for m, gen_image in enumerate(generated_images[:n_gen]):
        # Handle grayscale
        if gen_image.shape[-1] == 1:
            axes_gen[m].imshow(gen_image.squeeze(-1), cmap="gray")
        else:
            axes_gen[m].imshow(np.clip(gen_image, 0, 1))
        axes_gen[m].axis("off")
        axes_gen[m].set_title(f"Gen #{m+1}")

    fig_gen.suptitle("Generated Images from Model", fontsize=12)
    plt.tight_layout()
    fig_gen
    return


if __name__ == "__main__":
    app.run()
