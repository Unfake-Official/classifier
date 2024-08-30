from keras import losses, metrics, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
from classifier import Classifier
from tqdm import tqdm
import numpy as np
import os
import sys


class Trainer:
    def __init__(self, model: Classifier):
        self.model = model
        self.loss_obj = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.Adam()

        # loss metrics
        self.train_loss = metrics.MeanSquaredError(name='train_loss')
        self.train_accuracy = metrics.CategoricalAccuracy(
            name='train_accuracy')
        self.test_loss = metrics.MeanSquaredError(name='test_loss')
        self.test_accuracy = metrics.CategoricalAccuracy(
            name='test_accuracy')

        # metrics history
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

    def plot(self, output):

        fig = plt.figure(figsize=(10, 10))
        # n_rows, n_columns, index
        plt.subplot(2, 1, 1)
        plt.plot(self.train_accuracy_history, label='Training Accuracy')
        plt.plot(self.test_accuracy_history, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.test_loss_history, label='Validation Loss')
        plt.legend(loc='lower right')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Loss')

        fig.savefig(output, dpi=fig.dpi)

    def save_csv(self, output):
        np.savetxt(output, [p for p in
                                  zip(self.train_loss_history,
                                      self.train_accuracy_history,
                                      self.test_loss_history,
                                      self.test_accuracy_history)], delimiter=',', fmt='%s')

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            predictions = predictions['output_1']
            loss = self.loss_obj(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        self.train_loss(labels, predictions)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        predictions = predictions['output_1']

        self.test_loss(labels, predictions)
        self.test_accuracy(labels, predictions)

    def train(self, epochs: int, train_ds, test_ds, checkpoint_path: str, metrics_path: str, csv_path: str):
        print(f'Number of epochs: {epochs}')

        patience = 15
        wait = 0
        best = sys.maxsize

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}')

            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.test_loss.reset_state()
            self.test_accuracy.reset_state()

            print('Training')
            for train_images, train_labels in tqdm(train_ds):
                self.train_step(train_images, train_labels)

            print('Evaluating')
            for test_images, test_labels in tqdm(test_ds):
                self.test_step(test_images, test_labels)

            self.train_loss_history.append(self.train_loss.result())
            self.train_accuracy_history.append(self.train_accuracy.result())

            self.test_loss_history.append(self.test_loss.result())
            self.test_accuracy_history.append(self.test_accuracy.result())

            self.model.export(os.path.join(checkpoint_path))
            self.plot(metrics_path)
            self.save_csv(csv_path)

            test_loss = self.test_loss.result()

            wait += 1
            if test_loss < best:
                best = test_loss
                wait = 0
            if wait >= patience:
                break
