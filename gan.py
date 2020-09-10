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
from PIL import Image


# TODO: label smoothing broke these; they are still wrong
def accuracy_op(logits, y_smooth):
    y_hat = logits > 0
    y = y_smooth > 0.5
    indicators = y_hat == y
    return tf.reduce_mean(tf.cast(indicators, tf.float32))


def SaveAsGIF(images, path):
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)


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
        z = tf.random.normal(shape=(num_search_vectors, self.latent_dim))
        print(z.shape)
        generated_images = self.generator(z)
        predictions = self.discriminator(generated_images)
        nice_order = tf.argsort(predictions, direction="ASCENDING", axis=0)
        nicest_indices = nice_order[:num_nice_vectors]
        print(z)
        print(type(nicest_indices))
        print(z.shape)
        print(nicest_indices)
        print(nicest_indices.shape)
        return tf.gather(z, nicest_indices)

    def _build_generator(self):
        self.generator = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim,)),
                layers.Dense(7 * 7 * 128),
                # layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
                # layers.ReLU(),
                layers.Reshape((7, 7, 128)),
                # Upsample: 7x7 -> 14x14.
                layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"),
                # layers.BatchNormalization(),
                # layers.ReLU(),
                layers.LeakyReLU(alpha=0.2),
                # Upsample: 14x14 -> 28x28
                layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"),
                # layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
                # layers.ReLU(),
                # Reshape
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        self.generator.summary()
        self.g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

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
        self.d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
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

    def _train_step_discriminator_internal(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.discriminator(images)
            loss = self.loss_fn(labels, predictions)
        d_grads = tape.gradient(loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights)
        )
        acc = accuracy_op(predictions, labels)
        return loss, acc

    def _train_step_discriminator_generated(self, batch_size):
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(z)
        # Positive labels smoothing, Uniform [0.7, 1.2].
        labels = tf.random.uniform(shape=(batch_size, 1), minval=0.7, maxval=1.2)
        return self._train_step_discriminator_internal(generated_images, labels)

    def _train_step_discriminator_real(self, real_images):
        batch_size = tf.shape(real_images)[0]
        labels = tf.zeros((batch_size, 1))
        # Negative label smoothing Uniform, [0.0, 0.3].
        labels = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=0.3)
        return self._train_step_discriminator_internal(real_images, labels)

    def _train_step_discriminator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        d_loss_g, d_acc_g = self._train_step_discriminator_generated(batch_size)
        d_loss_r, d_acc_r = self._train_step_discriminator_real(real_images)
        d_loss = (d_loss_g + d_loss_r) / 2.0
        return d_loss_g, d_loss_r, d_acc_g, d_acc_r

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        d_loss_g, d_loss_r, d_acc_g, d_acc_r = self._train_step_discriminator(
            real_images
        )
        g_loss = self._train_step_generator(batch_size)
        return {
            "g_loss": g_loss,
            "d_loss_g": d_loss_g,
            "d_loss_r": d_loss_r,
            "d_acc_g": d_acc_g,
            "d_acc_r": d_acc_r,
        }

    def generate_and_save_images(self, z, path):
        generated_images = self.generator(z)
        generated_images *= 255
        generated_images.numpy()
        imgs = []
        for i in range(z.shape[0]):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            # img.save(path.format(i=i))
            imgs.append(img)
        SaveAsGIF(imgs, "interp.gif")


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

    most_recent_ckpt_path = "training/saved-model-most-recent.ckpt"

    load_weights = False
    if load_weights:
        dcgan.load_weights(most_recent_ckpt_path)

    if False:
        z_nice = dcgan.find_nice_latent_vectors(
            num_nice_vectors=10, num_search_vectors=2000
        )
        a = z_nice[0, :]
        b = z_nice[1, :]

        num_interp = 20
        a_to_b = np.zeros(shape=(num_interp, latent_dim))
        for i in range(num_interp):
            p = i / (num_interp - 1)
            a_to_b[i, :] = a + p * (b - a)
        b_to_a = np.flipud(a_to_b)
        interp_loop = np.vstack([a_to_b, b_to_a])
        dcgan.generate_and_save_images(interp_loop, "interp{i:03d}.png")

    train = True
    if train:
        epochs = 30

        # Save the most recent weights as a canonical place to load from.
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

        dcgan_monitor_callback = DCGANMonitor(num_img=3, latent_dim=latent_dim)

        dcgan.fit(
            mnist,
            epochs=epochs,
            callbacks=[ckpt_cb_each_epoch, ckpt_cb_most_recent, dcgan_monitor_callback],
        )
