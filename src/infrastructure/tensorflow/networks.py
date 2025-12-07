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

# -------------------------------------------------------------------------
# Copyright 2025 Thomas Boulier
#
# This file contains code derived from the implementation in:
# “Generative Deep Learning, 2nd Edition” by David Foster (O’Reilly).
# Original source code (Apache License 2.0) available at:
# https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition
#
# Modifications:
# - Adapted to fit the architecture and coding style of this project.
# - Added detailed docstrings and type annotations.
# - Encapsulated Keras imports to avoid global dependency.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# -------------------------------------------------------------------------


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
            # Concatenate upsampled features with skip connection
            x = layers.Concatenate()([x, skips.pop()])
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


def _compute_num_levels(image_size: int, min_resolution: int = 4) -> int:
    """
    Compute the number of downsampling levels based on image size.

    The number of levels is chosen so that the smallest feature map
    has at least `min_resolution` pixels on each side.

    Parameters
    ----------
    image_size : int
        Size of the input images (assumes square images).
    min_resolution : int, optional
        Minimum spatial resolution at the bottleneck. Default is 4.

    Returns
    -------
    int
        Number of downsampling levels (between 1 and 4).
    """
    num_levels = 0
    size = image_size
    while size >= min_resolution * 2 and num_levels < 4:
        size //= 2
        num_levels += 1
    return max(1, num_levels)


def get_unet(image_size: int, noise_embedding_size: int, num_channels: int = 1):
    """
    Construct a U-Net model for noise prediction in diffusion models.

    The U-Net architecture consists of:
    - An encoder path that progressively downsamples the input
    - A bottleneck with residual blocks
    - A decoder path that upsamples and combines with skip connections
    - Time embedding injection to condition on diffusion timestep

    The depth of the network adapts automatically to the image size to ensure
    that spatial dimensions remain compatible across skip connections.

    Parameters
    ----------
    image_size : int
        Size of the input images (assumes square images). Must be a power of 2
        or at least divisible by 2 for each downsampling level.
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
    Channel progression depends on depth:
    - 1 level: 32 → bottleneck → 32
    - 2 levels: 32 → 64 → bottleneck → 64 → 32
    - 3 levels: 32 → 64 → 96 → bottleneck → 96 → 64 → 32
    - 4 levels: 32 → 64 → 96 → 128 → bottleneck → 128 → 96 → 64 → 32
    """
    # Channel widths for each level (up to 4 levels)
    channel_widths = [32, 64, 96, 128]
    bottleneck_width = 128
    block_depth = 2

    # Compute number of levels based on image size
    num_levels = _compute_num_levels(image_size)

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
    for level in range(num_levels):
        width = channel_widths[level]
        x = DownBlock(width, block_depth=block_depth)([x, skips])

    # Bottleneck: process features at lowest resolution
    x = ResidualBlock(bottleneck_width)(x)
    x = ResidualBlock(bottleneck_width)(x)

    # Decoder: progressively upsample and decrease channels
    # Skip connections from encoder are concatenated at each level (LIFO order)
    for level in range(num_levels - 1, -1, -1):
        width = channel_widths[level]
        x = UpBlock(width, block_depth=block_depth)([x, skips])

    # Final projection: map to output noise prediction (same channels as input)
    # Initialize with zeros so initial predictions are near zero
    x = layers.Conv2D(num_channels, kernel_size=1, kernel_initializer="zeros")(x)

    unet = models.Model([noisy_images, noise_variances], x, name="unet")
    return unet
