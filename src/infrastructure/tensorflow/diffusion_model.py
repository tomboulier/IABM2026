"""
TensorFlow Diffusion Model Implementation.

This module implements a denoising diffusion probabilistic model (DDPM)
using TensorFlow/Keras, exposed through the domain `Model` interface.

Responsibilities
----------------
- Owns the U-Net backbone for noise prediction (from `networks.get_unet`)
- Implements the forward and reverse diffusion process
- Trains the model using TensorFlow (manual training loop)
- Bridges the domain `Dataset` protocol to TensorFlow via `TensorFlowDatasetAdapter`
- Returns generated images as numpy arrays (domain `Tensor` protocol)
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras

from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor
from src.domain.interfaces.model import Model
from src.infrastructure.tensorflow.dataset_adapter import TensorFlowDatasetAdapter
from src.infrastructure.tensorflow.networks import get_unet

# Keras shortcuts
layers = keras.layers
optimizers = keras.optimizers


def offset_cosine_diffusion_schedule(diffusion_times: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute noise and signal rates using an offset cosine schedule.

    Parameters
    ----------
    diffusion_times : tf.Tensor
        Diffusion timesteps in range [0, 1], where 0 is the start (clean images)
        and 1 is the end (pure noise). Shape: (batch_size, 1, 1, 1)

    Returns
    -------
    noise_rates : tf.Tensor
        Noise level at each timestep (sine of diffusion angle).
    signal_rates : tf.Tensor
        Signal level at each timestep (cosine of diffusion angle).
    """
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates


class TensorFlowDiffusionModel(Model):
    """
    Concrete TensorFlow implementation of the domain `Model` interface.

    This class encapsulates:
    - The U-Net network (`get_unet`)
    - The diffusion schedule and reverse diffusion process
    - A manual TensorFlow training loop
    - Conversions from domain `Dataset` to `tf.data.Dataset`
      and from TensorFlow tensors to numpy arrays (domain `Tensor`)

    Notes
    -----
    This class lives entirely in the infrastructure layer and is free to
    depend on TensorFlow/Keras. The domain only sees the abstract `Model`
    interface and remains framework-agnostic.
    """

    def __init__(
            self,
            image_size: int = 28,
            num_channels: int = 3,
            noise_embedding_size: int = 32,
            batch_size: int = 32,
            ema: float = 0.995,
            plot_diffusion_steps: int = 20,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            epochs: int = 1,
            load_weights_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the TensorFlow diffusion model.

        Parameters
        ----------
        image_size : int, optional
            Spatial size of square input images. Default is 28.
        num_channels : int, optional
            Number of image channels (1=grayscale, 3=RGB). Default is 3.
        noise_embedding_size : int, optional
            Dimension of the sinusoidal time embedding. Default is 32.
        batch_size : int, optional
            Training batch size. Default is 32.
        ema : float, optional
            Exponential moving average decay factor (0-1). Default is 0.995.
        plot_diffusion_steps : int, optional
            Number of reverse diffusion steps at generation time. Default is 20.
        learning_rate : float, optional
            AdamW learning rate. Default is 1e-4.
        weight_decay : float, optional
            AdamW weight decay. Default is 1e-4.
        epochs : int, optional
            Number of training epochs. Default is 1.
        load_weights_path : str, optional
            Optional path to weights file for the U-Net.
        """
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.ema = ema
        self.plot_diffusion_steps = plot_diffusion_steps
        self.epochs = epochs

        # Network and EMA copy
        self.network = get_unet(
            image_size=image_size,
            noise_embedding_size=noise_embedding_size,
            num_channels=num_channels,
        )
        self.ema_network = keras.models.clone_model(self.network)

        # Normalizer (channel-last)
        self.normalizer = layers.Normalization(axis=-1)

        # Optimizer and loss
        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_fn = keras.losses.MeanSquaredError()

        # Diffusion schedule
        self.diffusion_schedule = offset_cosine_diffusion_schedule

        # Optional weight loading
        if load_weights_path is not None:
            # Build the model implicitly by calling it once before loading weights
            dummy_images = tf.zeros(
                (1, image_size, image_size, num_channels), dtype=tf.float32
            )
            dummy_noise_var = tf.zeros((1, 1, 1, 1), dtype=tf.float32)
            _ = self.network([dummy_images, dummy_noise_var])
            self.network.load_weights(load_weights_path)
            # Mirror weights to EMA network
            for weight, ema_weight in zip(
                    self.network.weights, self.ema_network.weights
            ):
                ema_weight.assign(weight)

    # -------------------------------------------------------------------------
    # Public API (domain interface)
    # -------------------------------------------------------------------------

    def train(self, dataset: Dataset) -> None:
        """
        Train the diffusion model on a domain Dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset implementing the domain `Dataset` protocol.
        """
        # Basic compatibility checks with domain dataset
        if dataset.image_size != self.image_size:
            raise ValueError(
                f"Dataset image size {dataset.image_size} of dataset {dataset} "
                f"is not compatible with model image size {self.image_size}."
            )
        if dataset.num_channels != self.num_channels:
            raise ValueError(
                f"Dataset num_channels {dataset.num_channels} of dataset {dataset} "
                f"is not compatible with model num_channels {self.num_channels}."
            )

        # Convert domain Dataset to tf.data.Dataset
        tf_dataset = TensorFlowDatasetAdapter(
            dataset=dataset,
            batch_size=self.batch_size,
        ).to_tensorflow_dataset()

        # Adapt normalizer statistics on the full dataset
        self.normalizer.adapt(tf_dataset)

        # Training loop
        for _ in range(self.epochs):
            for images in tf_dataset:
                self._train_on_batch(images)

    def generate_images(self, n: int) -> Tensor:
        """
        Generate images using the trained model.

        Parameters
        ----------
        n : int
            Number of images to generate.

        Returns
        -------
        Tensor
            Generated images as numpy array in [0, 1], shape
            (n, image_size, image_size, num_channels).
        """
        initial_noise = tf.random.normal(
            shape=(n, self.image_size, self.image_size, self.num_channels),
            dtype=tf.float32,
        )
        generated = self._reverse_diffusion(
            initial_noise=initial_noise,
            diffusion_steps=self.plot_diffusion_steps,
        )
        generated = self._denormalize(generated)
        return generated.numpy()

    # -------------------------------------------------------------------------
    # Internal helpers: denoising and diffusion
    # -------------------------------------------------------------------------

    def _denormalize(self, images: tf.Tensor) -> tf.Tensor:
        """
        Convert normalized images back to [0, 1] range.

        Parameters
        ----------
        images : tf.Tensor
            Normalized images (zero mean, unit variance).

        Returns
        -------
        tf.Tensor
            Images clipped to [0, 1].
        """
        images = self.normalizer.mean + images * tf.sqrt(self.normalizer.variance)
        return tf.clip_by_value(images, 0.0, 1.0)

    def _denoise(
            self,
            noisy_images: tf.Tensor,
            noise_rates: tf.Tensor,
            signal_rates: tf.Tensor,
            training: bool,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Predict noise and reconstruct clean images from noisy inputs.

        Parameters
        ----------
        noisy_images : tf.Tensor
            Noisy images, shape (batch, H, W, C).
        noise_rates : tf.Tensor
            Noise rates, shape (batch, 1, 1, 1).
        signal_rates : tf.Tensor
            Signal rates, shape (batch, 1, 1, 1).
        training : bool
            Whether to use the main network (training=True)
            or the EMA network (training=False).

        Returns
        -------
        pred_noises : tf.Tensor
            Predicted noise.
        pred_images : tf.Tensor
            Predicted clean images.
        """
        network = self.network if training else self.ema_network

        pred_noises = network([noisy_images, noise_rates ** 2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def _reverse_diffusion(
            self,
            initial_noise: tf.Tensor,
            diffusion_steps: int,
    ) -> tf.Tensor:
        """
        Run the reverse diffusion process starting from pure noise.

        Parameters
        ----------
        initial_noise : tf.Tensor
            Initial Gaussian noise, shape (batch, H, W, C).
        diffusion_steps : int
            Number of denoising steps.

        Returns
        -------
        tf.Tensor
            Final predicted clean images (still normalized).
        """
        if diffusion_steps <= 0:
            return initial_noise

        num_images = tf.shape(initial_noise)[0]
        step_size = 1.0 / tf.cast(diffusion_steps, tf.float32)
        current_images = initial_noise
        pred_images = current_images

        for step in range(int(diffusion_steps)):
            diffusion_times = tf.ones(
                (num_images, 1, 1, 1), dtype=tf.float32
            ) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            pred_noises, pred_images = self._denoise(
                current_images,
                noise_rates,
                signal_rates,
                training=False,
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )

            current_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    # -------------------------------------------------------------------------
    # Internal helper: training on one batch
    # -------------------------------------------------------------------------

    def _train_on_batch(self, images: tf.Tensor) -> tf.Tensor:
        """
        Perform one training step on a batch of images.

        Parameters
        ----------
        images : tf.Tensor
            Batch of clean images.

        Returns
        -------
        tf.Tensor
            Scalar loss value.
        """
        batch_size = tf.shape(images)[0]

        # Normalize
        images = self.normalizer(images, training=True)

        # Sample noise and timesteps
        noises = tf.random.normal(
            shape=(batch_size, self.image_size, self.image_size, self.num_channels)
        )
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, _ = self._denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                training=True,
            )
            loss = self.loss_fn(noises, pred_noises)

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # EMA update
        for weight, ema_weight in zip(
                self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1.0 - self.ema) * weight)

        return loss
