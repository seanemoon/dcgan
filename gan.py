# Base model:
# https://keras.io/examples/generative/dcgan_overriding_train_step/
#
# Next steps are to make changes suggested here:
# https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
# TODO: use random normal weight initialziation (std=0.02)
#
# Fantastic high-quality resources:
# https://arxiv.org/pdf/1701.00160.pdf
# https://arxiv.org/pdf/1606.03498.pdf

import tensorflow as tf
import shutil
import numpy as np
import os
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


# This is bad but often recommended :C.
def smooth_labels_bad(y):
    batch_size = tf.shape(y)[0]
    y1_smooth = tf.random.uniform(shape=(batch_size, 1), minval=0.7, maxval=1.2)
    y0_smooth = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=0.3)
    return y * y1_smooth + (1 - y) * y0_smooth


# We should only smooth positive examples, and only for the discriminator.
def smooth_labels(y):
    batch_size = tf.shape(y)[0]
    y1_smooth = 0.75 * tf.ones(shape=(batch_size, 1))
    y0_smooth = tf.zeros(shape=(batch_size, 1))
    return y * y1_smooth + (1 - y) * y0_smooth


# Randomly flip `p` percent of labels.
def noisy_labels(y, p=0.05):
    batch_size = tf.shape(y)[0]
    y_flipped = 1 - y
    flip = tf.cast(tf.random.uniform(shape=(batch_size, 1)) < p, tf.float32)
    return flip * y_flipped + (1 - flip) * y


def SaveAsGIF(images, path):
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)


class DCGAN(tf.keras.Model):
    def __init__(self, latent_dim, num_classes):
        super(DCGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def compile(self):
        # super(DCGAN, self).compile()
        super(DCGAN, self).compile(
            metrics={"d_predictions": keras.metrics.SparseCategoricalAccuracy()}
        )
        self._build_generator()
        self._build_discriminator()

    def generate_images(self, num_images):
        z = tf.random.normal(shape=(num_images, self.latent_dim))
        generated_images = self.generator(z)
        generated_images += 1.0
        generated_images *= 127.5
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
                layers.LeakyReLU(alpha=0.2),
                # layers.ReLU(),
                layers.Reshape((7, 7, 128)),
                # Upsample: 7x7 -> 14x14.
                layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"),
                layers.BatchNormalization(),
                # layers.ReLU(),
                layers.LeakyReLU(alpha=0.2),
                # Upsample: 14x14 -> 28x28
                layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
                # layers.ReLU(),
                # Reshape
                layers.Conv2D(1, (7, 7), padding="same", activation="tanh"),
            ],
            name="generator",
        )
        self.generator.summary()
        self.g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    def _build_discriminator2(self):
        inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        self.d_features = layers.Flatten()(x)
        print("features shape: ", self.d_features.shape)
        self.d_predictions = layers.Dense(self.num_classes + 1, name="d_predictions")(
            self.d_features
        )
        self.discriminator = keras.Model(
            inputs=inputs,
            outputs={
                "d_predictions": self.d_predictions,
                "d_features": self.d_features,
            },
            name="discriminator",
        )
        self.d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.pred_loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.feat_loss_fn = keras.losses.MeanSquaredError()

    def _build_discriminator(self):
        if True:
            self._build_discriminator2()
        else:
            self.discriminator = keras.Sequential(
                [
                    keras.Input(shape=(28, 28, 1)),
                    # downsample 28x28 -> 14x14
                    layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2),
                    # downsample 14x14 -> 7x7
                    layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.2),
                    # FC
                    layers.GlobalMaxPooling2D(),
                    layers.Dense(self.num_classes + 1),
                ],
                name="discriminator",
            )
        self.discriminator.summary()
        self.d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    def _train_step_generator(self, batch_size, real_features):
        real_features = tf.identity(real_features)
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            outputs = self.discriminator(self.generator(z))
            predictions = outputs["d_predictions"]
            predictions_generated = tf.gather(
                predictions, indices=[self.num_classes], axis=1
            )
            misleading_labels = tf.zeros((batch_size, 1))
            deception_loss = keras.losses.BinaryCrossentropy(from_logits=True)(
                misleading_labels, predictions_generated
            )
            # TODO(seanrafferty): Add feature matching loss.
        g_grads = tape.gradient(deception_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        return deception_loss

    def _train_step_discriminator_internal(self, images, labels):
        with tf.GradientTape() as tape:
            outputs = self.discriminator(images)
            loss = self.pred_loss_fn(labels, outputs["d_predictions"])
        d_grads = tape.gradient(loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights)
        )
        accuracy = tf.keras.metrics.categorical_accuracy(
            labels, outputs["d_predictions"]
        )
        return loss, accuracy, outputs["d_features"]

    # Self-supervised training for the discriminator. We use the generator to
    # generate new images and expect the discriminator to classify them all as
    # generated.
    def _train_step_discriminator_generated(self, batch_size):
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(z)
        labels = tf.concat(
            [tf.zeros((batch_size, self.num_classes)), tf.ones((batch_size, 1))], axis=1
        )
        return self._train_step_discriminator_internal(generated_images, labels)

    # Supervised training for the discriminator. We use labeled real images and
    # expect the discriminator to correctly classify them as specific real
    # classes.
    def _train_step_discriminator_real(self, real_images, classes):
        labels = tf.one_hot(classes, depth=self.num_classes + 1)
        return self._train_step_discriminator_internal(real_images, labels)

    def _train_step_discriminator(self, real_images, classes):
        batch_size = tf.shape(real_images)[0]
        d_loss_g, d_acc_g, d_feat_g = self._train_step_discriminator_generated(
            batch_size
        )
        d_loss_r, d_acc_r, d_feat_r = self._train_step_discriminator_real(
            real_images, classes
        )
        return d_loss_g, d_loss_r, d_acc_r, d_acc_g, d_feat_r

    def train_step(self, train_data):
        real_images, classes = train_data

        batch_size = tf.shape(real_images)[0]
        d_loss_g, d_loss_r, d_acc_r, d_acc_g, d_feat_r = self._train_step_discriminator(
            real_images, classes
        )
        g_loss = self._train_step_generator(batch_size, d_feat_r)
        return {
            "g_loss": g_loss,
            "d_acc_r": d_acc_r,
            "d_acc_g": d_acc_g,
            "d_loss_g": d_loss_g,
            "d_loss_r": d_loss_r,
        }

    def call(self, real_images):
        return self.discriminator(real_images)

    def generate_and_save_images(self, z, path):
        generated_images = self.generator(z)
        generated_images += 1.0
        generated_images *= 127.5
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
        shutil.rmtree("generated_images")
        os.mkdir("generated_images")

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(z)
        generated_images += 1.0
        generated_images *= 127.5
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(
                "generated_images/epoch{epoch:03d}_img{i}.png".format(epoch=epoch, i=i)
            )


def prepare_mnist_split(x, y, batch_size=128):
    x = x.astype("float32") / 127.5 - 1.0
    x = np.reshape(x, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = (
        dataset.shuffle(buffer_size=1024)
        .batch(batch_size, drop_remainder=True)
        .prefetch(batch_size)
    )
    return dataset


def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    train_split = prepare_mnist_split(x_train, y_train)
    test_split = prepare_mnist_split(x_test, y_test)
    return train_split, test_split


if __name__ == "__main__":
    latent_dim = 128
    num_classes = 10

    dcgan = DCGAN(latent_dim, num_classes)
    dcgan.compile()

    mnist_train, mnist_test = prepare_mnist()

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
            mnist_train,
            validation_data=mnist_test,
            epochs=epochs,
            callbacks=[ckpt_cb_each_epoch, ckpt_cb_most_recent, dcgan_monitor_callback],
        )
