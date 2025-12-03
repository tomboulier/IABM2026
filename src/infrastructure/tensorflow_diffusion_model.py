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
    """
    Create a residual convolutional block factory that produces an apply(x) callable.
    
    Parameters:
        width (int): Number of output channels for the block.
    
    Returns:
        apply (callable): A function that accepts a 4D tensor `x` (NHWC) and returns `x` transformed by a residual block: input channels are matched to `width` (via identity or a 1x1 convolution), then the main path applies BatchNormalization (center=False, scale=False), a 3x3 convolution with Swish activation, a second 3x3 convolution without activation, and finally adds the residual to the main path.
    """
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
    """
    Create a downsampling block factory that returns a function applying repeated residual blocks and 2x2 average pooling.
    
    Parameters:
        width (int): Number of output channels for each ResidualBlock.
        block_depth (int): Number of consecutive ResidualBlock layers to apply before pooling.
    
    Returns:
        apply (callable): A function that expects a tuple `(x, skips)` where `x` is a feature tensor and `skips` is a list.
            The function applies `block_depth` ResidualBlock(width) layers to `x`, appends each block output to `skips`,
            then applies 2x2 average pooling and returns the pooled tensor.
    """
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    """
    Create an upsampling block factory that returns a function applying upsampling, skip-connection merging, and ResidualBlock processing.
    
    The returned function expects a tuple (x, skips), where x is the current feature tensor and skips is a list (or stack) of skip-connection tensors. It upsamples x by a factor of 2 (bilinear), then for block_depth iterations pops a skip tensor, crops the skip if its spatial dimensions exceed x's, concatenates the skip with x along the channel axis, and applies a ResidualBlock of the specified width to the result.
    
    Parameters:
        width (int): Number of output channels for each ResidualBlock in the upsampling path.
        block_depth (int): Number of residual blocks (and corresponding skip merges) to perform after upsampling.
    
    Returns:
        function: A callable that accepts (x, skips) and returns the processed feature tensor after upsampling, skip merging, and residual blocks.
    """
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
    """
    Compute sinusoidal positional embeddings for input diffusion times.
    
    Parameters:
        x (tf.Tensor): Tensor of scalar diffusion times or noise scales. The input's spatial/batch
            dimensions are preserved; embeddings vary along the last axis used for concatenation.
    
        noise_embedding_size (int): Total size of the output embedding; must be an even integer.
    
    Returns:
        tf.Tensor: Tensor of sinusoidal embeddings where the last concatenation axis (axis 3)
            has size `noise_embedding_size` and contains interleaved `sin` and `cos`
            components for log-spaced frequencies between 1 and 1000.
    """
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
    """
    Builds a U‑Net conditioned on sinusoidal noise embeddings for use in a diffusion model.
    
    Parameters:
        image_size (int): Spatial height and width of the model input and output.
        noise_embedding_size (int): Dimension of the sinusoidal noise embedding used to condition the network.
        num_channels (int): Number of channels for the input and output images.
    
    Returns:
        unet (keras.Model): Keras Model that accepts two inputs `(noisy_images, noise_variances)` and produces denoised images with shape `(image_size, image_size, num_channels)`.
    """
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
    Compute noise and signal rates from diffusion times using a cosine schedule with a small offset.
    
    Parameters:
        diffusion_times (tf.Tensor): Tensor of diffusion progression values, typically in the range [0, 1].
    
    Returns:
        tuple: (noise_rates, signal_rates) where both are tf.Tensor values computed from the cosine schedule:
            - noise_rates: sin of the scheduled angles (amount of noise).
            - signal_rates: cos of the scheduled angles (amount of signal).
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
        """
                 Initialize the diffusion model, build its U‑Net and EMA copy, and configure training and callbacks.
                 
                 Parameters:
                     image_size:
                         Spatial size of input images (images are square: image_size x image_size).
                     num_channels:
                         Number of image channels (e.g., 3 for RGB).
                     noise_embedding_size:
                         Dimensionality of the sinusoidal noise embedding used to condition the U‑Net.
                     batch_size:
                         Default training batch size the model will be used with.
                     ema:
                         Exponential moving average decay factor applied to the EMA network weights.
                     load_weights_path:
                         Optional filesystem path to pre-trained weights to load; if None, weights are initialized.
                     plot_diffusion_steps:
                         Number of reverse-diffusion steps used when producing example/generated images.
                     learning_rate:
                         Learning rate used to configure the optimizer during training.
                     weight_decay:
                         Weight decay (L2) applied by the optimizer.
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
        # Tracker metric must be created in __init__ (or build) to avoid
        # adding new variables after the model has been built.
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    def compile(self, **kwargs):
        """
        Configure the model for training by forwarding compilation arguments to the base Keras Model.
        
        Parameters:
            **kwargs: Keyword arguments accepted by keras.models.Model.compile (for example `optimizer`, `loss`, `metrics`) which are passed through to the superclass implementation.
        """
        super().compile(**kwargs)

    def build(self, input_shape):
        """
        Prepare the model for use by optionally loading weights and ensuring a dynamic batch dimension before calling the base build.
        
        Parameters:
            input_shape (tuple|Sequence|tf.TensorShape): Shape or shape-like object describing the model inputs. If a tuple with a fixed batch size in the first position, that fixed size will be replaced with `None` to allow dynamic batch sizes.
        
        """
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
        Ensure the model's default callbacks are attached and delegate to keras.Model.fit.
        
        If the caller provides callbacks, they are combined with the model's configured callbacks; if the provided value is None or not iterable it is replaced by the model's callbacks. The remaining fit arguments are passed through unchanged.
        
        Returns:
            History: Keras `History` object containing training metrics and state.
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
        Train the diffusion model on a TensorFlow dataset for a specified number of epochs.
        
        The method adapts the model's input normalizer to the provided dataset, compiles the model with the configured optimizer and loss, builds the model with the configured input shape, and runs training.
        
        Parameters:
            dataset (tf.data.Dataset): A TensorFlow dataset that yields image batches suitable for training.
            epochs (int): Number of epochs to train the model for (default: 1).
        """
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
        """
        Expose the model's tracked metrics for Keras monitoring.
        
        Returns:
            A list containing the Mean metric that tracks the model's noise prediction loss.
        """
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        """
        Restore images from the normalizer's original scale using stored mean and variance, then clip values to the [0, 1] range.
        
        Parameters:
            images: Tensor or array of normalized image values (zero-centered using the model's normalizer).
        
        Returns:
            Tensor of denormalized image values, clipped so every element is between 0.0 and 1.0.
        """
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """
        Predicts the noise component and corresponding denoised images for given noisy inputs, using the training network or the EMA network when not training.
        
        Parameters:
            noisy_images: Tensor of noisy images to denoise.
            noise_rates: Tensor of noise rate scalars applied to the inputs (same shape or broadcastable to noisy_images).
            signal_rates: Tensor of signal rate scalars used to reconstruct images (same shape or broadcastable to noisy_images).
            training (bool): If True, use the model's training network; if False, use the EMA (exponential moving average) network.
        
        Returns:
            (pred_noises, pred_images): 
                pred_noises — Tensor of noise predictions for each input.
                pred_images — Tensor of reconstructed (denoised) images computed from the predictions.
        """
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
        """
        Perform iterative reverse diffusion to denoise a batch of images starting from given noise.
        
        Parameters:
            initial_noise (tf.Tensor): Batch of noisy images to start from, shape (batch, height, width, channels).
            diffusion_steps (int): Number of discrete reverse diffusion steps to run; if <= 0, returns `initial_noise` unchanged.
        
        Returns:
            tf.Tensor: Final denoised images after running the reverse diffusion loop, same spatial/channel shape as `initial_noise`.
        """
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
        """
        Generate images by running the reverse diffusion process from an initial noise tensor.
        
        Parameters:
            num_images (int): Number of images to generate when `initial_noise` is not provided.
            diffusion_steps (int): Number of reverse diffusion steps to run.
            initial_noise (array-like or tf.Tensor, optional): If provided, must have shape (N, H, W, C); converted to float32 and used as the starting noise. When provided, `num_images` is ignored and N is used instead.
        
        Returns:
            tf.Tensor: Denormalized generated images with values clipped to [0, 1] and shape (N, H, W, C).
        """
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
        """
        Performs a single training step on a batch of images, updating the network weights, exponential moving average (EMA) weights, and tracked metrics.
        
        Parameters:
            images (tf.Tensor): Batch of input images with shape (batch_size, image_size, image_size, num_channels).
        
        Returns:
            dict: Mapping of metric names to their current values after the step.
        """
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
        """
        Performs a single evaluation step on a batch of images and updates the tracked noise loss metric.
        
        Parameters:
            images (tf.Tensor): Batch of input images, shape (batch_size, image_size, image_size, num_channels) or a shape broadcast-compatible with the model; dtype float32.
        
        Returns:
            dict: Mapping from metric names to their current scalar values (e.g., the noise loss).
        """
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
                 Adapter constructor that configures and instantiates a TensorFlow DiffusionModel for training and generation.
                 
                 Parameters:
                     image_size (int): Height/width of square images the model will process.
                     num_channels (int): Number of image channels (e.g., 1 for grayscale, 3 for RGB).
                     noise_embedding_size (int): Dimensionality of the sinusoidal noise embedding used by the U-Net.
                     batch_size (int): Default training batch size used when building the underlying model.
                     ema (float): Exponential moving average decay applied to the model weights for the EMA network.
                     plot_diffusion_steps (int): Number of diffusion steps used when generating example images for inspection.
                     learning_rate (float): Initial learning rate for the optimizer.
                     weight_decay (float): Weight decay (L2) applied by the optimizer.
                     epochs (int): Number of training epochs the adapter will run when training via its train() method.
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
        Train the adapter's underlying TensorFlow diffusion model using a domain Dataset.
        
        Converts the provided domain Dataset (which should yield images or (image, label) tuples) into a tf.data.Dataset with shape (image_size, image_size, num_channels), batches it using the adapter's configured batch size (dropping a final partial batch), adds standard prefetching, and then calls the underlying TensorFlow model's train method for the adapter's configured number of epochs.
        
        Parameters:
            dataset (Dataset): A domain-level Dataset implementing sequence access (len and indexing) that yields image arrays or (image, label) pairs. Images may be NumPy arrays, framework tensors (e.g., PyTorch), or array-like objects; they will be converted to float32 and reshaped to (H, W, C) as needed.
        """
        # Convert domain Dataset to TensorFlow dataset
        # Assuming the dataset already has the right format (PyTorch or TensorFlow)
        # For now, we'll need to handle this conversion carefully

        # Create a TensorFlow dataset from the protocol-based dataset
        def generator():
            """
            Yield preprocessed images from the outer `dataset` ready for batching by a TensorFlow Dataset.
            
            Each yielded item is a NumPy float32 image array with shape (H, W, C). The generator:
            - Accepts dataset items that are images or (image, label) pairs and extracts the image.
            - Converts PyTorch tensors to NumPy arrays if they expose `cpu()` and `numpy()` methods.
            - Ensures dtype is `float32`.
            - Reorders channels from (C, H, W) to (H, W, C) when the first dimension matches `self.num_channels`.
            - Adds a channel axis for single-channel images (2D arrays).
            
            Returns:
                numpy.ndarray: A single image array of shape (H, W, C) and dtype `float32` for each yield.
            """
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
        Generate a batch of images using the trained diffusion model.
        
        Returns:
            A NumPy array of generated images with shape (n, image_size, image_size, num_channels) and dtype float32; pixel values are in the range [0, 1].
        """
        # Generate using TensorFlow model
        generated_tf = self.tf_model.generate(
            num_images=n,
            diffusion_steps=self.plot_diffusion_steps
        )
        
        # Convert TensorFlow tensor to numpy array (boundary conversion)
        generated_np = generated_tf.numpy()
        
        return generated_np