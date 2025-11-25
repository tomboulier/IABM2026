import math

# from utils import display  # TODO: Create utils module or remove display functionality
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models
activations = keras.activations
metrics = keras.metrics
register_keras_serializable = keras.utils.register_keras_serializable
callbacks = keras.callbacks
optimizers = keras.optimizers

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            skip = skips.pop()
            # Handle shape mismatch by cropping skip connection to match x
            skip_shape = skip.shape
            x_shape = x.shape
            if skip_shape[1] != x_shape[1] or skip_shape[2] != x_shape[2]:
                # Crop skip to match x dimensions
                crop_h = skip_shape[1] - x_shape[1]
                crop_w = skip_shape[2] - x_shape[2]
                if crop_h > 0 or crop_w > 0:
                    skip = layers.Cropping2D(cropping=((0, crop_h), (0, crop_w)))(skip)
            x = layers.Concatenate()([x, skip])
            x = ResidualBlock(width)(x)
        return x

    return apply

@register_keras_serializable(package="diffusion")
def sinusoidal_embedding(x, noise_embedding_size: int):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            noise_embedding_size // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def get_unet(image_size: int, noise_embedding_size: int, num_channels: int = 1):
    noisy_images = layers.Input(shape=(image_size, image_size, num_channels))

    # Première projection des images
    x = layers.Conv2D(32, kernel_size=1)(noisy_images)

    # Embedding sinusoidal
    noise_variances = layers.Input(shape=(1, 1, 1))
    noise_embedding = layers.Lambda(
        sinusoidal_embedding,
        arguments={"noise_embedding_size": noise_embedding_size}
    )(noise_variances)
    noise_embedding = layers.UpSampling2D(
        size=image_size,
        interpolation="nearest"
    )(noise_embedding)

    # Projection de l’embedding pour matcher la largeur des features
    noise_embedding = layers.Conv2D(32, kernel_size=1)(noise_embedding)

    # Concaténation puis projection pour revenir à 32 canaux
    x = layers.Concatenate()([x, noise_embedding])
    x = layers.Conv2D(32, kernel_size=1)(x)

    skips = []
    x = DownBlock(32, block_depth=2)([x, skips])
    x = DownBlock(64, block_depth=2)([x, skips])
    x = DownBlock(96, block_depth=2)([x, skips])

    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)

    x = UpBlock(96, block_depth=2)([x, skips])
    x = UpBlock(64, block_depth=2)([x, skips])
    x = UpBlock(32, block_depth=2)([x, skips])

    x = layers.Conv2D(num_channels, kernel_size=1, kernel_initializer="zeros")(x)
    
    # Ensure output size matches input size exactly
    # Resize to original image_size in case of any size mismatch from convolutions
    x = layers.Resizing(image_size, image_size, interpolation="bilinear")(x)

    unet = models.Model([noisy_images, noise_variances], x, name="unet")
    return unet



def offset_cosine_diffusion_schedule(diffusion_times):
    """
    Implements the cosine diffusion schedule with small offset to avoid
    extremely small noise rates.
    """
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

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
                 weight_decay: float = 1e-4,):
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
        self.normalizer = layers.Normalization(axis=-1) # normalize images to mean 0 and variance 1 (axis=-1 for channels last)
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

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

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
        self.normalizer.adapt(dataset)
        self.compile(
            optimizer=optimizers.AdamW(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
            loss=keras.losses.MeanSquaredError(),
        )
        self.build((self.batch_size, self.image_size, self.image_size, self.num_channels))
        self.fit(dataset, epochs=epochs)

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates**2], training=training
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # Handle edge case where diffusion_steps <= 0: return initial noise directly
        if diffusion_steps <= 0:
            return initial_noise

        # Use tf.shape to handle symbolic shapes when running in graph mode
        num_images = tf.shape(initial_noise)[0]
        step_size = 1.0 / tf.cast(diffusion_steps, tf.float32)
        current_images = initial_noise
        # initialize pred_images to silence static analyzers (will be overwritten)
        pred_images = current_images
        # Use python range for step loop — diffusion_steps is expected to be int
        for step in range(int(diffusion_steps)):
            diffusion_times = tf.ones((num_images, 1, 1, 1), dtype=tf.float32) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, self.image_size, self.image_size, self.num_channels),
            )
            generated_images = self.reverse_diffusion(
                initial_noise, diffusion_steps
            )
        else:
            # ensure we work with tf tensors (no mixing numpy arrays + tf tensors)
            initial_noise = tf.convert_to_tensor(initial_noise, dtype=tf.float32)
            # keep num_images consistent with the provided initial_noise
            num_images = int(initial_noise.shape[0])
            generated_images = self.reverse_diffusion(
                initial_noise, diffusion_steps
                )
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # Determine runtime batch size from the input tensor
        batch_size = tf.shape(images)[0]
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, self.num_channels))

        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        batch_size = tf.shape(images)[0]
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, self.num_channels))
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}


# ==============================================================================
# ADAPTER LAYER: Implements domain Model interface
# ==============================================================================

import numpy as np

from src.domain.entities.dataset import Dataset
from src.domain.entities.tensor import Tensor
from src.domain.interfaces.model import Model


class TensorFlowDiffusionModelAdapter(Model):
    """
    Adapter that wraps the TensorFlow DiffusionModel to implement the domain's Model interface.
    
    This class handles boundary conversions:database
    - Accepts framework-agnostic Dataset protocol
    - Returns numpy arrays (Tensor protocol) instead of TensorFlow tensors
    
    This is how Clean Architecture works: domain defines the interface,
    infrastructure implements it with framework-specific code and adapts at boundaries.
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
        Initialize TensorFlow diffusion model adapter.
        
        Parameters match the underlying TensorFlow model.
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
        Train the model on a dataset.
        
        Adapts from domain Dataset protocol to TensorFlow dataset.
        """
        # Convert domain Dataset to TensorFlow dataset
        # Assuming the dataset already has the right format (PyTorch or TensorFlow)
        # For now, we'll need to handle this conversion carefully

        # Create a TensorFlow dataset from the protocol-based dataset
        def generator():
            for i in range(len(dataset)):
                item = dataset[i]
                # Support dataset returning (image, label) or image alone
                if isinstance(item, tuple) or isinstance(item, list):
                    image = item[0]
                else:
                    image = item

                # If it's a torch tensor, move to cpu and convert to numpy
                # Avoid importing torch as a hard dependency at module import time
                # by checking for common torch attributes.
                if hasattr(image, "cpu") and hasattr(image, "numpy"):
                    try:
                        image = image.cpu().numpy()
                    except Exception:
                        # Fallback: try just .numpy()
                        image = image.numpy()

                # Ensure numpy array dtype float32
                image = np.array(image, dtype=np.float32)

                # Transpose from (C, H, W) to (H, W, C) if needed
                if image.ndim == 3 and image.shape[0] == self.num_channels and image.shape[0] < image.shape[1]:
                    image = np.transpose(image, (1, 2, 0))

                # If a single-channel image is missing the channel axis, add it
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
        
        Returns numpy arrays (implements Tensor protocol) instead of TensorFlow tensors.
        This is the boundary conversion - TensorFlow → numpy.
        """
        # Generate using TensorFlow model
        generated_tf = self.tf_model.generate(
            num_images=n,
            diffusion_steps=self.plot_diffusion_steps
        )
        
        # Convert TensorFlow tensor to numpy array (boundary conversion)
        generated_np = generated_tf.numpy()
        
        return generated_np
