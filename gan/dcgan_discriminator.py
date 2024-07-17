from keras import layers, Model, ops

class DCGAN_Discriminator(Model):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch1 = layers.BatchNormalization()
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.2)

        self.conv2 = layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch2 = layers.BatchNormalization()
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.2)

        self.conv3 = layers.Conv2D(64 * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batch3 = layers.BatchNormalization()
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.2)

        self.conv4 = layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid', use_bias=False)
        self.flatten = layers.Flatten()
        self.sigmoid = layers.Activation('sigmoid')

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
        x = self.flatten(x)
        x = ops.mean(x, axis=1)
        x = self.sigmoid(x)

        return x
