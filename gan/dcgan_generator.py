from keras import layers, Model

class DCGAN_Generator(Model):
    def __init__(self):
        super(DCGAN_Generator, self).__init__()

        self.deconv1 = layers.Conv2DTranspose(256 * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.deconv2 = layers.Conv2DTranspose(256 * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.deconv3 = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

        self.deconv4 = layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.tanh = layers.Activation('tanh')

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
