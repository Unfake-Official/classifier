'''
Reference: https://www.geeksforgeeks.org/generative-adversarial-network-gan/
'''
import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.d1 = tf.keras.layers.Dense(8 * 8 * 128, activation='relu')
        self.reshape = tf.keras.layers.Reshape((8, 8, 128))
        self.upsample1 = tf.keras.layers.UpSampling2D(size=(2,2))
        self.conv1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=0.78)
        self.upsample2 = tf.keras.layers.UpSampling2D(size=(2,2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(momentum=0.75)
        self.conv3 = tf.keras.layers.Conv2D(3, (3,3), activation='tanh')

    def call(self, x):
        x = self.d1(x)
        x = self.reshape(1)
        x = self.upsample1(1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        return x
