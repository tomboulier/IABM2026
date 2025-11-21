import logging
from src.domain.entities.datasets import load_dataset
from src.domain.entities.metrics import resnet_features_mean_square_centroid, frechet_inception_distance
from src.domain.entities.models import diffusion_model

from src.infrastructure.configuration import ExperimentConfiguration

logger = logging.getLogger(__name__)

def run_experiment(config: ExperimentConfiguration):
    """
    Runs the variability and similarity experiment based on the configuration.
    
    Args:
        config: The experiment configuration.
    """
    datasets = config.datasets
    max_samples = config.max_samples
    image_size = config.image_size

    logger.info("Starting MedMNIST Variability & Similarity Experiment")

    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name}...")
        
        try:
            # 1. Load Dataset
            dataset = load_dataset(dataset_name, max_samples=max_samples, image_size=image_size)
            logger.info(f"Loaded {dataset_name} with {len(dataset)} samples.")

            # 2. Compute Variability
            variability_score = resnet_features_mean_square_centroid(dataset)
            logger.info(f"Variability (Mean Squared Distance to Centroid) for {dataset_name}: {variability_score:.4f}")

            # 3. Train Model (Mock)
            logger.info(f"Training diffusion model on {dataset_name}...")
            diffusion_model.train(dataset)

            # 4. Generate Images
            logger.info(f"Generating images for {dataset_name}...")
            generated_images = diffusion_model.generate_images(n=100) # Generate 100 for quick test
            
            # 5. Compute Similarity
            similarity_score = frechet_inception_distance(dataset, generated_images)
            logger.info(f"Similarity (FID-like) for {dataset_name}: {similarity_score:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")

    logger.info("Experiment completed.")
