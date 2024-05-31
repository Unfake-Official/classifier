import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from dcgan_generator import DCGAN_Generator
from dcgan_discriminator import DCGAN_Discriminator
from classifier import Classifier


class Trainer:
    def __init__(self, classifier: Classifier, generator: DCGAN_Generator, discriminator: DCGAN_Discriminator):
        self.classifier = classifier
        self.generator = generator
        self.discriminator = discriminator

        # optimizers
        self.optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, weight_decay=1e-3)
        self.optimizer_generator = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, weight_decay=1e-3)
        self.optimizer_classifier = tf.keras.optimizers.Adam()

        # losses
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.criterion = tf.keras.losses.CategoricalCrossentropy()

        # adversarial weight
        self.adversarial_weight = 0.1

        # loss metrics
        self.train_loss_generator = tf.keras.metrics.MeanSquaredError(name='train_loss_generator')
        self.train_loss_discriminator = tf.keras.metrics.MeanSquaredError(name='train_loss_discriminator')
        self.train_loss_classifier = tf.keras.metrics.MeanSquaredError(name='train_loss_classifier')

        self.train_accuracy_generator = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_generator')
        self.train_accuracy_discriminator = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_discriminator')
        self.train_accuracy_classifier = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_classifier')

        self.test_loss_generator = tf.keras.metrics.MeanSquaredError(name='test_loss_generator')
        self.test_loss_discriminator = tf.keras.metrics.MeanSquaredError(name='test_loss_discriminators')
        self.test_loss_classifier = tf.keras.metrics.MeanSquaredError(name='test_loss_classifier')

        self.test_accuracy_generator = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_generator')
        self.test_accuracy_discriminator = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_discriminator')
        self.test_accuracy_classifier = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_classifier')

        # metrics history
        self.train_loss_history_generator = []
        self.train_loss_history_discriminator = []
        self.train_loss_history_classifier = []

        self.train_accuracy_history_generator = []
        self.train_accuracy_history_discriminator = []
        self.train_accuracy_history_classifier = []

        self.test_loss_history_generator = []
        self.test_loss_history_discriminator = []
        self.test_loss_history_classifier = []

        self.test_accuracy_history_generator = []
        self.test_accuracy_history_discriminator = []
        self.test_accuracy_history_classifier = []

    def plot(self, output):

        fig = plt.figure(figsize=(10, 10))
        # n_rows, n_columns, index
        plt.subplot(2, 1, 1)

        plt.plot(self.train_accuracy_history_generator, label='Generator Training Accuracy')
        plt.plot(self.test_accuracy_history_generator, label='Generator Validation Accuracy')
        plt.plot(self.train_accuracy_history_discriminator, label='Discriminator Training Accuracy')
        plt.plot(self.test_accuracy_history_discriminator, label='Discriminator Validation Accuracy')
        plt.plot(self.train_accuracy_history_classifier, label='Classifier Training Accuracy')
        plt.plot(self.test_accuracy_history_classifier, label='Classifier Validation Accuracy')

        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(self.train_loss_history_generator, label='Generator Training Loss')
        plt.plot(self.test_loss_history_generator, label='Generator Validation Loss')
        plt.plot(self.train_loss_history_discriminator, label='Discriminator Training Loss')
        plt.plot(self.test_loss_history_discriminator, label='Discriminator Validation Loss')
        plt.plot(self.train_loss_history_classifier, label='Classifier Training Loss')
        plt.plot(self.test_loss_history_classifier, label='Classifier Validation Loss')

        plt.legend(loc='lower right')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Loss')

        fig.savefig(output, dpi=fig.dpi)

    def train_step(self, images, labels, batch_size):
        true_label = tf.ones(batch_size)
        fake_label = tf.ones(batch_size)

        # generate random noise to feed generator
        noise = tf.random.normal([batch_size, 256, 1, 1])
        fake_images = self.generator(noise, training=True)

        with tf.GradientTape() as tape:
            # train discriminator on real images
            predictions_discriminator_real = self.discriminator(images, training=True)
            loss_discriminator_real = self.loss(true_label, predictions_discriminator_real)

            # train discriminator on fake images
            predictions_discriminator_fake = self.discriminator(fake_images, training=True)
            loss_discriminator_fake = self.loss(fake_label, predictions_discriminator_fake)

        gradients = tape.gradient(loss_discriminator_real, self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        gradients = tape.gradient(loss_discriminator_fake, self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # train generator
        with tf.GradientTape() as tape:
            loss_generator = self.loss(true_label, predictions_discriminator_fake)

        gradients = tape.gradient(loss_generator, self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # train classifier on real data
        with tf.GradientTape() as tape:
            predictions_classifier_real = self.classifier(images, training=True)
            real_classifier_loss = self.criterion(labels, predictions_classifier_real)

        gradients = tape.gradient(real_classifier_loss, self.classifier.trainable_variables)
        self.optimizer_classifier.apply_gradients(zip(gradients, self.classifier.trainable_variables))

        with tf.GradientTape() as tape:
            # Update classifier on fake data
            predictions_classifier_fake = self.classifier(fake_images, training=True)
            predicted_labels = tf.argmax(predictions_classifier_fake, axis=1)

            confidence_thresh = 0.2

            # Pseudo labeling threshold
            probs = tf.nn.softmax(predictions_discriminator_fake , axis=1)
            most_likely_probs = tf.gather_nd(probs, tf.stack((tf.range(tf.size(predicted_labels)), predicted_labels), axis=1))

            to_keep = tf.greater(most_likely_probs, confidence_thresh)
            to_keep_indices = tf.where(to_keep)[:, 0]  # Get indices where condition is True

            if tf.reduce_sum(tf.cast(to_keep, tf.int32)) != 0:
                # Compute fake classifier loss only if there are samples to keep
                fake_classifier_loss = self.criterion(tf.gather(predictions_discriminator_fake, to_keep_indices),
                                                    tf.gather(predicted_labels, to_keep_indices)) * self.adversarial_weight

        gradients = tape.gradient(fake_classifier_loss, self.classifier.trainable_variables)
        self.optimizer_classifier.apply_gradients(zip(gradients, self.classifier.trainable_variables))

        self.train_loss_discriminator(true_label, predictions_discriminator_real)
        self.train_loss_discriminator(fake_label, predictions_discriminator_fake)

        self.train_accuracy_discriminator(true_label, predictions_discriminator_real)
        self.train_accuracy_discriminator(fake_label, predictions_discriminator_fake)

        self.train_loss_generator(true_label, predictions_discriminator_fake)
        self.train_accuracy_generator(true_label, predictions_discriminator_fake)

        self.train_loss_classifier(labels, predictions_classifier_real)
        self.train_loss_classifier(labels, predictions_classifier_fake)

        self.train_accuracy_classifier(labels, predictions_classifier_real)
        self.train_accuracy_classifier(labels, predictions_classifier_fake)

    def test_step(self, images, labels, batch_size):
        true_label = tf.ones(batch_size)
        fake_label = tf.ones(batch_size)

        # generate random noise to feed generator
        noise = tf.random.normal([batch_size, 256, 1, 1])
        fake_images = self.generator(noise, training=False)

        predictions_discriminator_real = self.discriminator(images, training=False)
        predictions_discriminator_fake = self.discriminator(fake_images, training=False)

        predictions_classifier_real = self.classifier(images, training=False)
        predictions_classifier_fake = self.classifier(fake_images, training=False)

        self.train_loss_discriminator(true_label, predictions_discriminator_real)
        self.train_loss_discriminator(fake_label, predictions_discriminator_fake)

        self.train_accuracy_discriminator(true_label, predictions_discriminator_real)
        self.train_accuracy_discriminator(fake_label, predictions_discriminator_fake)

        self.train_loss_generator(true_label, predictions_discriminator_fake)
        self.train_accuracy_generator(true_label, predictions_discriminator_fake)

        self.train_loss_classifier(labels, predictions_classifier_real)
        self.train_loss_classifier(labels, predictions_classifier_fake)

        self.train_accuracy_classifier(labels, predictions_classifier_real)
        self.train_accuracy_classifier(labels, predictions_classifier_fake)

    def train(self, epochs: int, train_ds, test_ds, generator_checkpoint_path: str, discriminator_checkpoint_path: str, classifier_checkpoint_path: str, metrics_path: str):
        print(f'Number of epochs: {epochs}')

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}')

            self.train_loss_generator.reset_states()
            self.train_accuracy_generator.reset_states()
            self.test_loss_generator.reset_states()
            self.test_accuracy_generator.reset_states()

            self.train_loss_discriminator.reset_states()
            self.train_accuracy_discriminator.reset_states()
            self.test_loss_discriminator.reset_states()
            self.test_accuracy_discriminator.reset_states()

            self.train_loss_classifier.reset_states()
            self.train_accuracy_classifier.reset_states()
            self.test_loss_classifier.reset_states()
            self.test_accuracy_classifier.reset_states()

            print('Training')
            for train_images, train_labels in tqdm(train_ds):
                batch_size = len(train_labels)
                self.train_step(train_images, train_labels, batch_size)

            print('Evaluating')
            for test_images, test_labels in tqdm(test_ds):
                batch_size = len(test_labels)
                self.test_step(test_images, test_labels, batch_size)

            self.train_loss_history_generator.append(self.train_loss_generator.result())
            self.train_accuracy_history_generator.append(self.train_accuracy_generator.result())
            self.test_loss_history_generator.append(self.test_loss_generator.result())
            self.test_accuracy_history_generator.append(self.test_accuracy_generator.result())

            self.train_loss_history_discriminator.append(self.train_loss_generator.result())
            self.train_accuracy_history_discriminator.append(self.train_accuracy_generator.result())
            self.test_loss_history_discriminator.append(self.test_loss_generator.result())
            self.test_accuracy_history_discriminator.append(self.test_accuracy_generator.result())

            self.train_loss_history_classifier.append(self.train_loss_generator.result())
            self.train_accuracy_history_classifier.append(self.train_accuracy_generator.result())
            self.test_loss_history_classifier.append(self.test_loss_generator.result())
            self.test_accuracy_history_classifier.append(self.test_accuracy_generator.result())

            self.generator.save(os.path.join(generator_checkpoint_path))
            self.discriminator.save(os.path.join(discriminator_checkpoint_path))
            self.classifier.save(os.path.join(classifier_checkpoint_path))

            self.plot(metrics_path)
