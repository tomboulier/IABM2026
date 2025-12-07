import numpy as np
import tensorflow as tf

from src.domain.entities.dataset import Dataset


class TensorFlowDatasetAdapter:
    """
    Adapter to convert domain Dataset protocol to TensorFlow tf.data.Dataset.

    This adapter handles:
    - Conversion from domain Dataset to tf.data.Dataset
    - Framework conversions (PyTorch tensors, numpy arrays → TensorFlow tensors)
    - Channel ordering adjustments (C,H,W) → (H,W,C)

    This ensures the domain layer remains independent of TensorFlow specifics.
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int
                 ):
        """
        Initialize the TensorFlow dataset adapter.

        Parameters
        ----------
        dataset : Dataset
            The domain Dataset protocol instance.
        batch_size : int
            The batch size for the TensorFlow dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size

    @staticmethod
    def generator(dataset: Dataset):
        """
        Generator that yields images from the dataset in TensorFlow format.

        Handles conversion from various input formats (PyTorch, numpy) to
        standardized (H, W, C) float32 numpy arrays.
        """
        for i in range(len(dataset)):
            item = dataset[i]

            # Extract image from dataset item
            # Support dataset returning (image, label) tuple or image alone
            if isinstance(item, tuple) or isinstance(item, list):
                image = item[0]  # First element is the image
            else:
                image = item  # Item is the image itself

            # Handle PyTorch tensor conversion
            # Check for PyTorch tensor attributes without importing torch
            # (avoids hard dependency at module level)
            if hasattr(image, "cpu") and hasattr(image, "numpy"):
                try:
                    # Move to CPU first (if on GPU), then convert to numpy
                    image = image.cpu().numpy()
                except Exception:
                    # Fallback: tensor might already be on CPU
                    image = image.numpy()

            # Ensure numpy array with float32 dtype
            # This handles remaining TensorFlow tensors and raw numpy arrays
            image = np.array(image, dtype=np.float32)

            # Handle channel ordering: (C, H, W) → (H, W, C)
            # PyTorch uses channels-first, TensorFlow uses channels-last
            # Check if first dimension is channel count and smaller than spatial dims
            if image.ndim == 3 and image.shape[0] == dataset.num_channels and image.shape[0] < image.shape[1]:
                image = np.transpose(image, (1, 2, 0))

            # Handle grayscale images without explicit channel dimension
            # Convert (H, W) → (H, W, 1)
            if image.ndim == 2:
                image = np.expand_dims(image, -1)

            yield image

    def to_tensorflow_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Convert domain Dataset to TensorFlow tf.data.Dataset.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the dataset each epoch. Default is True.

        Returns
        -------
        tf.data.Dataset
            Batched and optimized TensorFlow dataset.
        """
        # Create TensorFlow dataset from generator
        tf_dataset = tf.data.Dataset.from_generator(
            lambda: self.generator(self.dataset),
            output_signature=tf.TensorSpec(
                shape=(self.dataset.image_size, self.dataset.image_size, self.dataset.num_channels),
                dtype=tf.float32
            )
        )

        # Cache dataset in memory after first epoch
        # This avoids re-running the generator and conversions each epoch
        tf_dataset = tf_dataset.cache()

        # Shuffle for better training (reshuffle each epoch)
        if shuffle:
            buffer_size = min(len(self.dataset), 10000)  # Cap buffer to avoid OOM
            tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

        # Batch the dataset. Use drop_remainder=True to ensure consistent batch sizes
        # (avoids a final smaller batch which can cause shape mismatches during XLA compilation)
        tf_dataset = tf_dataset.batch(self.batch_size, drop_remainder=True)

        # Prefetch to overlap data loading with training
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        return tf_dataset
