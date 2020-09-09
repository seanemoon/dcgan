# Base model:
# https://keras.io/examples/generative/dcgan_overriding_train_step/
#
# Next steps are to make changes suggested here:
# https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
# TODO: add batch normalization
# TODO: use tanh in generator output
# TODO: use random normal weight initialziation (std=0.02)
# TODO: tweak adam optimizer (lr=2e-4, b1=0.5)
# TODO: increase batch size from 64 to 128

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations


class DCGAN(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DCGAN, self).__init__()
        self.latent_dim = latent_dim

    def compile(self):
        super(DCGAN, self).compile()
        self._build_generator()
        self._build_discriminator()

    def generate_images(self, num_images):
        z = tf.random.normal(shape=(num_images, self.latent_dim))
        generated_images = self.generator(z)
        generated_images *= 255
        generated_images.numpy()
        return generated_images

    def save(self, prefix):
        self.generator.save("{prefix}_generator".format(prefix=prefix))
        self.discriminator.save("{prefix}_discriminator".format(prefix=prefix))

    def find_nice_latent_vectors(self, num_nice_vectors, num_search_vectors):
        assert num_nice_vectors <= num_search_vectors
        z = tf.random.noraml(shape(num_search_vectors, self.latent_dim))
        generated_images = self.generator(z)
        predictions = self.discriminator(generated_images)
        predictions.numpy()
        nice_order = np.argsort(predictions)
        nicest_indices = nice_order[:num_nice_vectors]
        return z[nicest_indices, :]

    def _build_generator(self):
        self.generator = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim,)),
                layers.Dense(7 * 7 * 128),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7, 128)),
                # Upsample: 7x7 -> 14x14.
                layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                # Upsample: 14x14 -> 28x28
                layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                # Reshape
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        self.generator.summary()
        self.g_optimizer = keras.optimizers.Adam(learning_rate=3e-4)

    def _build_discriminator(self):
        self.discriminator = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                # downsample 28x28 -> 14x14
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                # downsample 14x14 -> 7x7
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                # FC
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )
        self.discriminator.summary()
        self.d_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    def _train_step_generator(self, batch_size):
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(z))
            loss = self.loss_fn(misleading_labels, predictions)
        g_grads = tape.gradient(loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        return loss

    def _train_step_discriminator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        # Generate images.
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(z)
        # Combine the generated and real images.
        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels -- important trick!
        # TODO(seanrafferty): why?
        combined_labels += 0.05 * tf.random.uniform(tf.shape(combined_labels))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            loss = self.loss_fn(combined_labels, predictions)
        d_grads = tape.gradient(loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights)
        )
        return loss

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        d_loss = self._train_step_discriminator(real_images)
        g_loss = self._train_step_generator(batch_size)
        return {"d_loss": d_loss, "g_loss": g_loss}


class DCGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(z)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(
                "generated_images/epoch{epoch:03d}_img{i}.png".format(epoch=epoch, i=i)
            )


def prepare_mnist():
    batch_size = 128
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)
    return dataset


if __name__ == "__main__":
    latent_dim = 128

    dcgan = DCGAN(latent_dim)
    dcgan.compile()

    mnist = prepare_mnist()

    # Save the most recent weights as a canonical place to load from.
    most_recent_ckpt_path = "training/saved-model-most-recent.ckpt"
    ckpt_cb_most_recent = keras.callbacks.ModelCheckpoint(
        filepath=most_recent_ckpt_path,
        save_weights_only=True,
        verbose=1,
    )

    # Save weights after each epoc so we can see how the model evolves.
    ckpt_cb_each_epoch = keras.callbacks.ModelCheckpoint(
        filepath="training/saved-model-{epoch:02d}.ckpt",
        save_weights_only=True,
        verbose=1,
    )

    load_weights = False
    if load_weights:
        dcgan.load_weights(most_recent_ckpt_path)

    dcgan_monitor_callback = DCGANMonitor(num_img=3, latent_dim=latent_dim)

    epochs = 30
    train = True
    if train:
        dcgan.fit(
            mnist,
            epochs=epochs,
            callbacks=[ckpt_cb_each_epoch, ckpt_cb_most_recent, dcgan_monitor_callback],
        )
