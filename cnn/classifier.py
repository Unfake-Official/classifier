from keras import layers, Model


class Classifier(Model):
    def __init__(self, trainable=True, *args, **kwargs):
        super(Classifier, self).__init__(trainable=trainable, *args, **kwargs)

        # convolution + feature extraction
        self.conv1 = layers.Conv2D(16, (3, 3))
        self.batch_norm1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.max_pool1 = layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = layers.Conv2D(32, (3, 3))
        self.batch_norm2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.max_pool2 = layers.MaxPool2D(pool_size=2, strides=2)
        self.conv3 = layers.Conv2D(64, (3, 3))
        self.batch_norm3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')
        self.max_pool3 = layers.MaxPool2D(pool_size=2, strides=2)
        self.conv4 = layers.Conv2D(64, (3, 3))
        self.batch_norm4 = layers.BatchNormalization()
        self.relu4 = layers.Activation('relu')
        self.max_pool4 = layers.MaxPool2D(pool_size=2, strides=2)

        self.dropout = layers.Dropout(0.5)

        # dense layer + output
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(1024, activation='relu')

        self.dropout2 = layers.Dropout(0.5)

        self.d2 = layers.Dense(2, activation='softmax')

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        x = self.max_pool4(x)
        if training:
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.d1(x)
        if training:
            x = self.dropout2(x)
        x = self.d2(x)
        return x
