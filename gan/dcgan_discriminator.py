import tensorflow as tf

class DCGAN_Discriminator(tf.keras.Model):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv2 = tf.keras.layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.leaky_relu2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv3 = tf.keras.layers.Conv2D(64 * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.leaky_relu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
            
        self.conv4 = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        
        return x
