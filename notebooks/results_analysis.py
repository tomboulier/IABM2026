import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
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
    return Path, df, notebook_dir, results_path


@app.cell
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
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create scatter plot
    scatter = ax.scatter(
        df["variability"],
        df["similarity"],
        s=100,
        alpha=0.7,
        c=range(len(df)),
        cmap="viridis",
    )

    # Add labels for each point
    for idx, row in df.iterrows():
        ax.annotate(
            row["dataset"].replace("MNIST", ""),
            (row["variability"], row["similarity"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xlabel("Variability (MSD to Centroid)", fontsize=12)
    ax.set_ylabel("Similarity (FID)", fontsize=12)
    ax.set_title("Dataset Variability vs Model Similarity", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Summary Statistics

    | Metric | Min | Max | Mean | Std |
    |--------|-----|-----|------|-----|
    | Variability | {df['variability'].min():.2f} | {df['variability'].max():.2f} | {df['variability'].mean():.2f} | {df['variability'].std():.2f} |
    | Similarity (FID) | {df['similarity'].min():.2f} | {df['similarity'].max():.2f} | {df['similarity'].mean():.2f} | {df['similarity'].std():.2f} |
    """)
    return


if __name__ == "__main__":
    app.run()
