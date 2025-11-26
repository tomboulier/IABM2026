"""
TensorFlow Diffusion Model Implementation.

This module implements a denoising diffusion probabilistic model (DDPM) using TensorFlow/Keras.
The architecture consists of:
- A U-Net backbone with residual blocks for noise prediction
- Sinusoidal time embeddings for diffusion timestep conditioning
- Exponential moving average (EMA) for stable generation
- An adapter layer to integrate with the domain model interface

The diffusion process gradually adds noise to images (forward diffusion),
and the model learns to reverse this process (reverse diffusion) to generate new samples.
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor
from src.domain.interfaces.model import Model

# from utils import display  # TODO: Create utils module or remove display functionality

# Keras module shortcuts for cleaner code
layers = keras.layers
models = keras.models
activations = keras.activations
metrics = keras.metrics
register_keras_serializable = keras.utils.register_keras_serializable
callbacks = keras.callbacks
optimizers = keras.optimizers


def ResidualBlock(width):
    """
    Create a residual block with batch normalization and skip connections.

    Residual blocks help with gradient flow in deep networks by adding
    skip connections that bypass the convolutional layers.

    Parameters
    ----------
    width : int
        Number of output channels for the convolutional layers.

    Returns
    -------
    callable
        A function that applies the residual block to an input tensor.
    """
    def apply(x):
        input_width = x.shape[3]
        # Adjust residual connection if input width doesn't match output width
        if input_width == width:
            residual = x
        else:
            # Project input to match output width using 1x1 convolution
            residual = layers.Conv2D(width, kernel_size=1)(x)

        # Normalize (scale=False for efficiency)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        # First convolution with swish activation (smooth, non-monotonic)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=activations.swish
        )(x)
        # Second convolution (no activation here - added after residual connection)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        # Add skip connection for residual learning
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    """
    Create a downsampling block for the U-Net encoder path.

    This block applies multiple residual blocks and then downsamples the feature maps.
    Skip connections are stored for later use in the decoder (UpBlock).

    Parameters
    ----------
    width : int
        Number of channels for the convolutional layers.
    block_depth : int
        Number of residual blocks to apply before downsampling.

    Returns
    -------
    callable
        A function that applies the downsampling block to (x, skips) tuple.
    """
    def apply(x):
        x, skips = x
        # Apply residual blocks and store outputs for skip connections
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)  # Save for decoder
        # Downsample by factor of 2 using average pooling
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    """
    Create an upsampling block for the U-Net decoder path.

    This block upsamples the feature maps and combines them with skip connections
    from the encoder via concatenation, then applies residual blocks.

    Parameters
    ----------
    width : int
        Number of channels for the convolutional layers.
    block_depth : int
        Number of residual blocks to apply after upsampling and concatenation.

    Returns
    -------
    callable
        A function that applies the upsampling block to (x, skips) tuple.
    """
    def apply(x):
        x, skips = x
        # Upsample by factor of 2 using bilinear interpolation
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)

        for _ in range(block_depth):
            # Retrieve skip connection from encoder (LIFO order)
            skip = skips.pop()

            # Handle shape mismatch by cropping skip connection to match x
            # This can occur due to rounding in pooling/upsampling operations
            skip_shape = skip.shape
            x_shape = x.shape
            if skip_shape[1] != x_shape[1] or skip_shape[2] != x_shape[2]:
                # Crop skip to match x dimensions (remove excess pixels)
                crop_h = skip_shape[1] - x_shape[1]
                crop_w = skip_shape[2] - x_shape[2]
                if crop_h > 0 or crop_w > 0:
                    skip = layers.Cropping2D(cropping=((0, crop_h), (0, crop_w)))(skip)

            # Concatenate upsampled features with skip connection
            x = layers.Concatenate()([x, skip])
            # Apply residual block to process combined features
            x = ResidualBlock(width)(x)
        return x

    return apply


@register_keras_serializable(package="diffusion")
def sinusoidal_embedding(x, noise_embedding_size: int):
    """
    Create sinusoidal position embeddings for diffusion timesteps.

    This function encodes continuous time values into high-dimensional vectors
    using sine and cosine functions at different frequencies. This helps the model
    distinguish between different noise levels during the diffusion process.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor containing time values (typically noise variance levels).
        Shape: (batch_size, 1, 1, 1)
    noise_embedding_size : int
        Dimension of the output embedding vector (must be even).

    Returns
    -------
    tf.Tensor
        Sinusoidal embeddings of shape (batch_size, 1, 1, noise_embedding_size).

    Notes
    -----
    The frequencies range exponentially from 1 to 1000, providing a rich
    representation that captures both fine and coarse temporal information.
    """
    # Create exponentially spaced frequencies from 1 to 1000
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            noise_embedding_size // 2,
        )
    )
    # Convert frequencies to angular speeds (radians per unit)
    angular_speeds = 2.0 * math.pi * frequencies

    # Create embeddings by concatenating sin and cos at different frequencies
    # This provides a unique encoding for each timestep
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def get_unet(image_size: int, noise_embedding_size: int, num_channels: int = 1):
    """
    Construct a U-Net model for noise prediction in diffusion models.

    The U-Net architecture consists of:
    - An encoder path that progressively downsamples the input
    - A bottleneck with residual blocks
    - A decoder path that upsamples and combines with skip connections
    - Time embedding injection to condition on diffusion timestep

    Parameters
    ----------
    image_size : int
        Size of the input images (assumes square images).
    noise_embedding_size : int
        Dimension of the sinusoidal time embedding vector.
    num_channels : int, optional
        Number of channels in the input images (1 for grayscale, 3 for RGB).
        Default is 1.

    Returns
    -------
    keras.Model
        A Keras model that takes [noisy_images, noise_variances] as input
        and outputs predicted noise of the same shape as the input images.

    Notes
    -----
    The model uses:
    - 32 → 64 → 96 channels in encoder
    - 128 channels in bottleneck
    - 96 → 64 → 32 channels in decoder
    - Skip connections between encoder and decoder at each resolution
    """
    # Input for noisy images at current diffusion step
    noisy_images = layers.Input(shape=(image_size, image_size, num_channels))

    # Initial projection: map input to 32 feature channels
    x = layers.Conv2D(32, kernel_size=1)(noisy_images)

    # Create time embedding from noise variance level
    noise_variances = layers.Input(shape=(1, 1, 1))
    noise_embedding = layers.Lambda(
        sinusoidal_embedding,
        arguments={"noise_embedding_size": noise_embedding_size}
    )(noise_variances)
    # Upsample embedding to match spatial dimensions of feature maps
    noise_embedding = layers.UpSampling2D(
        size=image_size,
        interpolation="nearest"
    )(noise_embedding)

    # Project time embedding to match feature channel width (32)
    noise_embedding = layers.Conv2D(32, kernel_size=1)(noise_embedding)

    # Inject time information by concatenating with image features
    x = layers.Concatenate()([x, noise_embedding])
    # Project back to 32 channels after concatenation
    x = layers.Conv2D(32, kernel_size=1)(x)

    # Encoder: progressively downsample and increase channels
    # Skip connections are stored in the 'skips' list
    skips = []
    x = DownBlock(32, block_depth=2)([x, skips])  # Output: 32 channels, H/2 x W/2
    x = DownBlock(64, block_depth=2)([x, skips])  # Output: 64 channels, H/4 x W/4
    x = DownBlock(96, block_depth=2)([x, skips])  # Output: 96 channels, H/8 x W/8

    # Bottleneck: process features at lowest resolution
    x = ResidualBlock(128)(x)  # 128 channels, H/8 x W/8
    x = ResidualBlock(128)(x)

    # Decoder: progressively upsample and decrease channels
    # Skip connections from encoder are concatenated at each level
    x = UpBlock(96, block_depth=2)([x, skips])  # Output: 96 channels, H/4 x W/4
    x = UpBlock(64, block_depth=2)([x, skips])  # Output: 64 channels, H/2 x W/2
    x = UpBlock(32, block_depth=2)([x, skips])  # Output: 32 channels, H x W

    # Final projection: map to output noise prediction (same channels as input)
    # Initialize with zeros so initial predictions are near zero
    x = layers.Conv2D(num_channels, kernel_size=1, kernel_initializer="zeros")(x)

    # Ensure output size matches input size exactly
    # (handles any rounding issues from pooling/upsampling)
    x = layers.Resizing(image_size, image_size, interpolation="bilinear")(x)

    unet = models.Model([noisy_images, noise_variances], x, name="unet")
    return unet


def offset_cosine_diffusion_schedule(diffusion_times):
    """
    Compute noise and signal rates using an offset cosine schedule.

    This schedule smoothly interpolates between clean images (high signal, low noise)
    and pure noise (low signal, high noise) using a cosine function. The offset
    prevents extreme values that could cause numerical instability.

    Parameters
    ----------
    diffusion_times : tf.Tensor
        Diffusion timesteps in range [0, 1], where 0 is the start (clean images)
        and 1 is the end (pure noise). Shape: (batch_size, 1, 1, 1)

    Returns
    -------
    noise_rates : tf.Tensor
        Noise level at each timestep (sine of diffusion angle).
        Range: approximately [0.02, 0.95]
    signal_rates : tf.Tensor
        Signal level at each timestep (cosine of diffusion angle).
        Range: approximately [0.02, 0.95]

    Notes
    -----
    The cosine schedule provides a smooth, gradual transition that improves
    training stability compared to linear schedules. The relationship
    signal_rates² + noise_rates² ≈ 1 is maintained throughout the process.
    """
    # Define signal rate bounds to avoid extreme values
    min_signal_rate = 0.02  # Minimum signal to avoid complete noise
    max_signal_rate = 0.95  # Maximum signal to ensure some noise is always present

    # Convert signal rates to angles (in radians)
    start_angle = tf.acos(max_signal_rate)  # Angle at t=0 (mostly signal)
    end_angle = tf.acos(min_signal_rate)  # Angle at t=1 (mostly noise)

    # Linearly interpolate angles based on diffusion time
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    # Compute signal and noise rates using trigonometric functions
    # This ensures they sum to approximately 1 (in squared space)
    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates


# Callbacks
# TODO: Uncomment ImageGenerator when utils.display is available
# class ImageGenerator(callbacks.Callback):
#     """Generates and saves sample images after every epoch."""
#     def __init__(self, num_img, plot_diffusion_steps: int):
#         self.num_img = num_img
#         self.plot_diffusion_steps = plot_diffusion_steps
# 
#     def on_epoch_end(self, epoch, logs=None):
#         generated_images = self.model.generate(
#             num_images=self.num_img,
#             diffusion_steps=self.plot_diffusion_steps,
#         ).numpy()
#         display(
#             generated_images,
#             save_to=f"./output/generated_img_{epoch:03d}.png",
#         )

class DiffusionModel(models.Model):
    def __init__(self,
                 image_size: int,
                 num_channels: int,
                 noise_embedding_size: int,
                 batch_size: int,
                 ema: float = 0.995,
                 load_weights_path: str = None,
                 plot_diffusion_steps: int = 20,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4, ):
        """Implements a diffusion model with a U-Net architecture.
        

        Parameters
        ----------
        image_size: int
            The size of the input images (image_size x image_size).
        num_channels: int
            The number of channels in the input images (e.g., 3 for RGB, 1 for grayscale).
        noise_embedding_size: int
            The size of the noise embedding vector.
        batch_size: int
            The batch size for training.
        ema: float
            The exponential moving average factor for the model weights.
        load_weights_path: str
            Path to load pre-trained weights. If None, weights are initialized randomly.
        plot_diffusion_steps: int
            The number of diffusion steps to use when generating sample images.
        learning_rate: float
            The learning rate for the optimizer.
        weight_decay: float
            The weight decay for the optimizer.
        """
        super().__init__()
        # parameters
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.ema = ema
        self.plot_diffusion_steps = plot_diffusion_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # model components
        self.normalizer = layers.Normalization(
            axis=-1)  # normalize images to mean 0 and variance 1 (axis=-1 for channels last)
        self.network = get_unet(image_size, noise_embedding_size, num_channels=num_channels)
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

        # callbacks
        self.callbacks_list = [
            # ImageGenerator(num_img=5, plot_diffusion_steps=self.plot_diffusion_steps),  # Commented out - needs utils.display
            callbacks.ModelCheckpoint(
                filepath="./checkpoint/ckpt.weights.h5",
                save_weights_only=True,
                save_freq="epoch",
                verbose=0,
            ),
            callbacks.TensorBoard(log_dir="./logs"),
        ]

        # load weights if a path is provided
        self.load_weights_path = load_weights_path
        # Tracker metric must be created in __init__ (or build) to avoid
        # adding new variables after the model has been built.
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    def compile(self, **kwargs):
        super().compile(**kwargs)

    def build(self, input_shape):
        if self.load_weights_path is not None:
            self.built = True
            self.load_weights(self.load_weights_path)
        # Allow dynamic batch dimension to avoid requiring a fixed batch size
        # from the TensorFlow dataset (which may use None as first dim).
        if isinstance(input_shape, tuple) and input_shape[0] is not None:
            # replace fixed batch with None for flexibility
            input_shape = (None,) + tuple(input_shape[1:])
        super().build(input_shape)

    def fit(self, *args, **kwargs):
        """
        Wrapper around keras.Model.fit that ensures our callbacks are attached.
        Accepts the full flexible signature (x=None, y=None, batch_size=None, epochs=1, ...)
        so it won't raise signature-mismatch warnings in static analysis.
        """
        provided_callbacks = kwargs.get("callbacks")
        if provided_callbacks is None:
            kwargs["callbacks"] = list(self.callbacks_list)
        else:
            # Merge while avoiding mutating the original sequence
            try:
                kwargs["callbacks"] = list(provided_callbacks) + list(self.callbacks_list)
            except TypeError:
                # If provided_callbacks isn't iterable, replace it
                kwargs["callbacks"] = list(self.callbacks_list)

        return super().fit(*args, **kwargs)

    def train(self, dataset: tf.data.Dataset, epochs: int = 1):
        """
        Train the diffusion model on a dataset.

        This method adapts the normalizer to the dataset statistics, compiles
        the model with AdamW optimizer, and trains for the specified epochs.

        Parameters
        ----------
        dataset : tf.data.Dataset
            A batched TensorFlow dataset containing images.
        epochs : int, optional
            Number of training epochs. Default is 1.
        """
        # Adapt normalizer to compute dataset mean and variance
        self.normalizer.adapt(dataset)
        # Configure optimizer and loss
        self.compile(
            optimizer=optimizers.AdamW(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
            loss=keras.losses.MeanSquaredError(),
        )
        # Build model with expected input shape
        self.build((self.batch_size, self.image_size, self.image_size, self.num_channels))
        # Train the model
        self.fit(dataset, epochs=epochs)

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        """
        Convert normalized images back to the original value range [0, 1].

        Reverses the normalization applied during training by the normalizer layer.

        Parameters
        ----------
        images : tf.Tensor
            Normalized images with zero mean and unit variance.

        Returns
        -------
        tf.Tensor
            Images clipped to the range [0, 1].
        """
        # Reverse normalization: denormalized = mean + normalized * std
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        # Clip to valid pixel range
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """
        Predict and separate noise from noisy images.

        Uses the U-Net to predict the noise component, then recovers the
        denoised image using the predicted noise and known signal/noise rates.

        Parameters
        ----------
        noisy_images : tf.Tensor
            Images with added noise. Shape: (batch_size, H, W, C)
        noise_rates : tf.Tensor
            Noise level for each image. Shape: (batch_size, 1, 1, 1)
        signal_rates : tf.Tensor
            Signal level for each image. Shape: (batch_size, 1, 1, 1)
        training : bool
            Whether to use the training network or EMA network.

        Returns
        -------
        pred_noises : tf.Tensor
            Predicted noise component.
        pred_images : tf.Tensor
            Predicted denoised images.

        Notes
        -----
        During training, uses the standard network. During inference/generation,
        uses the EMA network for more stable predictions.
        The denoising formula: pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        """
        # Select network based on training mode
        if training:
            network = self.network
        else:
            network = self.ema_network  # Use EMA for stable generation

        # Predict noise using U-Net (conditioned on noise variance)
        pred_noises = network(
            [noisy_images, noise_rates ** 2], training=training
        )
        # Recover clean images using predicted noise
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """
        Iteratively denoise pure noise to generate images.

        This implements the reverse diffusion process: starting from pure noise,
        the model gradually removes noise over multiple steps to produce a clean image.

        Parameters
        ----------
        initial_noise : tf.Tensor
            Pure random noise to start the generation process.
            Shape: (num_images, image_size, image_size, num_channels)
        diffusion_steps : int
            Number of denoising steps. More steps generally produce better quality
            but take longer. Typical values: 20-1000.

        Returns
        -------
        tf.Tensor
            Generated images (still normalized, needs denormalization).

        Notes
        -----
        Each step predicts the noise at the current timestep and computes
        a less noisy version for the next timestep. The process moves backward
        through time from t=1 (pure noise) to t=0 (clean image).
        """
        # Handle edge case where diffusion_steps <= 0: return initial noise directly
        if diffusion_steps <= 0:
            return initial_noise

        # Use tf.shape to handle symbolic shapes when running in graph mode
        num_images = tf.shape(initial_noise)[0]
        step_size = 1.0 / tf.cast(diffusion_steps, tf.float32)
        current_images = initial_noise
        # initialize pred_images to silence static analyzers (will be overwritten)
        pred_images = current_images

        # Iteratively denoise: move backward through diffusion timesteps
        # Use python range for step loop — diffusion_steps is expected to be int
        for step in range(int(diffusion_steps)):
            # Current diffusion time (1.0 at start, approaching 0.0 at end)
            diffusion_times = tf.ones((num_images, 1, 1, 1), dtype=tf.float32) - step * step_size
            # Get noise/signal rates for current timestep
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            # Predict noise and clean image at current timestep
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            # Compute next (less noisy) timestep
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            # Create next (less noisy) image by blending predicted clean image with noise
            current_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        # Return final denoised prediction
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        """
        Generate new images using the trained diffusion model.

        This is the main generation method that creates images by running
        the reverse diffusion process and denormalizing the results.

        Parameters
        ----------
        num_images : int
            Number of images to generate.
        diffusion_steps : int
            Number of denoising steps for generation. More steps = better quality.
        initial_noise : tf.Tensor or np.ndarray, optional
            Custom initial noise to use. If None, random noise is generated.
            Shape: (num_images, image_size, image_size, num_channels)

        Returns
        -------
        tf.Tensor
            Generated images in range [0, 1].
            Shape: (num_images, image_size, image_size, num_channels)
        """
        if initial_noise is None:
            # Generate random Gaussian noise as starting point
            initial_noise = tf.random.normal(
                shape=(num_images, self.image_size, self.image_size, self.num_channels),
            )
            # Run reverse diffusion to denoise
            generated_images = self.reverse_diffusion(
                initial_noise, diffusion_steps
            )
        else:
            # Use provided initial noise (convert to TensorFlow tensor if needed)
            # ensure we work with tf tensors (no mixing numpy arrays + tf tensors)
            initial_noise = tf.convert_to_tensor(initial_noise, dtype=tf.float32)
            # keep num_images consistent with the provided initial_noise
            num_images = int(initial_noise.shape[0])
            generated_images = self.reverse_diffusion(
                initial_noise, diffusion_steps
            )
        # Denormalize to [0, 1] range for display
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        """
        Custom training step for one batch of images.

        This implements the diffusion model training objective:
        1. Add random noise to images at random timesteps
        2. Train the network to predict the added noise
        3. Update weights using gradient descent
        4. Update EMA network for stable generation

        Parameters
        ----------
        images : tf.Tensor
            Batch of clean training images.
            Shape: (batch_size, image_size, image_size, num_channels)

        Returns
        -------
        dict
            Dictionary of metric names to values (e.g., {'n_loss': 0.042}).

        Notes
        -----
        This method is called automatically by Keras during training.
        The forward diffusion equation: noisy = signal_rate * clean + noise_rate * noise
        """
        # Determine runtime batch size from the input tensor
        batch_size = tf.shape(images)[0]
        # Normalize images to zero mean, unit variance
        images = self.normalizer(images, training=True)
        # Sample random Gaussian noise
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, self.num_channels))

        # Sample random diffusion timesteps for each image in batch
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        # Get corresponding noise and signal rates from schedule
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # Create noisy images by blending clean images with noise (forward diffusion)
        noisy_images = signal_rates * images + noise_rates * noises

        # Train the network to predict the noise component
        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            # Compute MSE loss between true noise and predicted noise
            noise_loss = self.loss(noises, pred_noises)  # used for training

        # Compute gradients and update weights
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        # Update loss metric for logging
        self.noise_loss_tracker.update_state(noise_loss)

        # Update EMA network weights for stable generation
        # EMA helps reduce variance in generated images
        for weight, ema_weight in zip(
                self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # Return metrics for Keras logging
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        """
        Custom test/validation step for one batch of images.

        Similar to train_step but without weight updates. Used for validation
        during training to monitor generalization.

        Parameters
        ----------
        images : tf.Tensor
            Batch of validation images.
            Shape: (batch_size, image_size, image_size, num_channels)

        Returns
        -------
        dict
            Dictionary of metric names to values.

        Notes
        -----
        This method is called automatically by Keras during validation.
        """
        # Determine batch size
        batch_size = tf.shape(images)[0]
        # Normalize (without updating normalizer statistics)
        images = self.normalizer(images, training=False)
        # Sample noise and diffusion times
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, self.num_channels))
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        # Get noise/signal rates
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # Create noisy images
        noisy_images = signal_rates * images + noise_rates * noises
        # Predict noise (no gradient tracking)
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        # Compute loss
        noise_loss = self.loss(noises, pred_noises)
        # Update metric
        self.noise_loss_tracker.update_state(noise_loss)

        # Return metrics for logging
        return {m.name: m.result() for m in self.metrics}


class TensorFlowDiffusionModelAdapter(Model):
    """
    Adapter that wraps TensorFlow DiffusionModel to implement domain Model interface.

    This adapter class is a key component of Clean Architecture, acting as a boundary
    between the infrastructure layer (TensorFlow-specific code) and the domain layer
    (framework-agnostic business logic).

    Responsibilities
    ----------------
    - Converts domain Dataset protocol to TensorFlow tf.data.Dataset
    - Handles framework conversions (PyTorch tensors, numpy arrays → TensorFlow)
    - Returns numpy arrays (Tensor protocol) instead of TensorFlow tensors
    - Manages channel ordering (C,H,W) → (H,W,C) conversion

    This ensures the domain layer remains independent of TensorFlow specifics.

    Attributes
    ----------
    image_size : int
        Size of square input images.
    num_channels : int
        Number of image channels (1 for grayscale, 3 for RGB).
    epochs : int
        Number of training epochs.
    plot_diffusion_steps : int
        Number of diffusion steps for generation.
    tf_model : DiffusionModel
        The underlying TensorFlow diffusion model instance.

    Examples
    --------
    >>> adapter = TensorFlowDiffusionModelAdapter(image_size=28, num_channels=1)
    >>> adapter.train(my_dataset)
    >>> generated = adapter.generate_images(n=10)  # Returns numpy array
    """

    def __init__(self,
                 image_size: int = 28,
                 num_channels: int = 3,  # Default to RGB
                 noise_embedding_size: int = 32,
                 batch_size: int = 32,
                 ema: float = 0.995,
                 plot_diffusion_steps: int = 20,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 epochs: int = 1):
        """
        Initialize the TensorFlow diffusion model adapter.

        Creates a wrapped DiffusionModel instance with specified hyperparameters.

        Parameters
        ----------
        image_size : int, optional
            Size of square input images. Default is 28.
        num_channels : int, optional
            Number of image channels (1=grayscale, 3=RGB). Default is 3.
        noise_embedding_size : int, optional
            Dimension of sinusoidal time embeddings. Default is 32.
        batch_size : int, optional
            Training batch size. Default is 32.
        ema : float, optional
            Exponential moving average decay for weights (0-1). Default is 0.995.
        plot_diffusion_steps : int, optional
            Number of diffusion steps for image generation. Default is 20.
        learning_rate : float, optional
            AdamW optimizer learning rate. Default is 1e-4.
        weight_decay : float, optional
            AdamW optimizer weight decay (L2 regularization). Default is 1e-4.
        epochs : int, optional
            Number of training epochs. Default is 1.
        """
        self.image_size = image_size
        self.num_channels = num_channels
        self.epochs = epochs
        self.plot_diffusion_steps = plot_diffusion_steps

        # Create the actual TensorFlow model
        self.tf_model = DiffusionModel(
            image_size=image_size,
            num_channels=num_channels,
            noise_embedding_size=noise_embedding_size,
            batch_size=batch_size,
            ema=ema,
            plot_diffusion_steps=plot_diffusion_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

    def train(self, dataset: Dataset):
        """
        Train the diffusion model on a domain Dataset.

        This method performs the critical boundary conversion from the domain's
        framework-agnostic Dataset protocol to TensorFlow's tf.data.Dataset.
        It handles multiple input formats (PyTorch tensors, numpy arrays) and
        normalizes them to the expected TensorFlow format.

        Parameters
        ----------
        dataset : Dataset
            A dataset implementing the Dataset protocol (__len__ and __getitem__).
            Can contain PyTorch tensors, numpy arrays, or TensorFlow tensors.

        Notes
        -----
        Conversion steps performed:
        1. Extract images from dataset items (handles tuple or single item)
        2. Convert PyTorch tensors to numpy (.cpu().numpy())
        3. Ensure float32 dtype
        4. Transpose from (C,H,W) to (H,W,C) if needed
        5. Add channel dimension for grayscale images if missing
        6. Create batched tf.data.Dataset with prefetching

        The final batch uses drop_remainder=True to avoid shape mismatches
        during XLA compilation.
        """
        # Convert domain Dataset to TensorFlow dataset
        # This generator function handles the conversion from domain protocol to TensorFlow

        def generator():
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
                if image.ndim == 3 and image.shape[0] == self.num_channels and image.shape[0] < image.shape[1]:
                    image = np.transpose(image, (1, 2, 0))

                # Handle grayscale images without explicit channel dimension
                # Convert (H, W) → (H, W, 1)
                if image.ndim == 2:
                    image = np.expand_dims(image, -1)

                yield image

        # Create TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(
                shape=(self.image_size, self.image_size, self.num_channels),
                dtype=tf.float32
            )
        )
        # Batch the dataset. Use drop_remainder=True to ensure consistent batch sizes
        # (avoids a final smaller batch which can cause shape mismatches during XLA compilation)
        tf_dataset = tf_dataset.batch(self.tf_model.batch_size, drop_remainder=True)
        # Add simple prefetch to improve input pipeline performance
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        # Train the TensorFlow model
        self.tf_model.train(tf_dataset, epochs=self.epochs)

    def generate_images(self, n: int) -> Tensor:
        """
        Generate images using the trained model.
        
        This method performs boundary conversion from TensorFlow tensors to numpy
        arrays, implementing the domain's Tensor protocol. This maintains the
        separation between infrastructure (TensorFlow) and domain layers.

        Parameters
        ----------
        n : int
            Number of images to generate.

        Returns
        -------
        Tensor (numpy.ndarray)
            Generated images as numpy array in range [0, 1].
            Shape: (n, image_size, image_size, num_channels)

        Notes
        -----
        The returned numpy array implements the Tensor protocol defined in the
        domain layer, allowing the domain to remain framework-agnostic.

        Examples
        --------
        >>> adapter = TensorFlowDiffusionModelAdapter(image_size=28, num_channels=1)
        >>> adapter.train(dataset)
        >>> images = adapter.generate_images(n=5)
        >>> images.shape
        (5, 28, 28, 1)
        """
        # Generate using TensorFlow model
        generated_tf = self.tf_model.generate(
            num_images=n,
            diffusion_steps=self.plot_diffusion_steps
        )

        # Convert TensorFlow tensor to numpy array (boundary conversion)
        # This is the key adapter responsibility: converting infrastructure types
        # to domain types (numpy arrays implementing Tensor protocol)
        generated_np = generated_tf.numpy()

        return generated_np
