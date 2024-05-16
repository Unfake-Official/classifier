import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = tf.keras.layers.Conv2DTranspose(64 * 4, (4, 4), strides=(1, 1), padding='valid', use_bias=False)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.deconv2 = tf.keras.layers.Conv2DTranspose(64 * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.deconv3 = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        self.deconv4 = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.tanh = tf.keras.layers.Activation('tanh')
    
    def call(self, x):
        x = self.deconv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.deconv4(x)
        x = self.tanh(x)
        
        return x