"""
infrastructure.tensorflow.networks
=================================

Module providing small, focused TensorFlow / Keras building blocks and a
U-Net factory used by the project's diffusion models.

This module exposes:
- block factories: `ResidualBlock`, `DownBlock`, `UpBlock` used to compose
  the U-Net encoder/decoder.
- `sinusoidal_embedding`: a registered Keras function producing time-step
  embeddings for conditioning the network on diffusion timesteps.
- `get_unet`: a convenience factory that builds and returns a Keras
  Model implementing the U-Net used to predict noise.

The implementation keeps Keras imports local to the module so other parts
of the codebase do not pay the TensorFlow import cost when they don't need
it.

Examples
--------
>>> from infrastructure.tensorflow.networks import get_unet
>>> model = get_unet(image_size=28, noise_embedding_size=64, num_channels=1)

Notes
-----
This docstring follows the numpydoc style. The module is intentionally
small and focused on model construction; dataset-specific concerns such as
`image_size` and `num_channels` should live in dataset loader classes, not
here (so the networks remain reusable across datasets).
"""

import math

import tensorflow as tf
from tensorflow import keras

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
