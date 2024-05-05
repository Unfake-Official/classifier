'''
Reference: https://www.geeksforgeeks.org/generative-adversarial-network-gan/
'''
import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), strides=2, activation='leaky_relu')
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), strides=2, activation='leaky_relu')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=0.82)
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.conv3 = tf.keras.layers.Conv2D(128, (3,3), strides=2, activation='leaky_relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(momentum=0.82)
        self.dropout3 = tf.keras.layers.Dropout(0.25)
        self.conv3 = tf.keras.layers.Conv2D(128, (3,3), strides=2, activation='leaky_relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(momentum=0.82)
        self.dropout3 = tf.keras.layers.Dropout(0.25)
        self.conv4 = tf.keras.layers.Conv2D(256, (3,3), strides=1, activation='leaky_relu')
        self.batch_norm3 = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.dropout4 = tf.keras.layers.Dropout(0.25)
        
        self.flatten = tf.keras.layers.Flatten(),
        self.d1 = tf.keras.layers.Dense(256 * 5 * 5, activation='leaky_relu')
        self.d2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = self.dropout4(x)
        
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
