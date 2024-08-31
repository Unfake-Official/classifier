from keras import layers, Model

# CNN model sketch:


class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()

        # convolution + feature extraction
        self.conv1 = layers.Conv2D(16, (3, 3))
        self.relu1 = layers.Activation('relu')
        self.max_pool1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.dropout1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv2D(32, (3, 3))
        self.relu2 = layers.Activation('relu')
        self.max_pool2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.dropout2 = layers.Dropout(0.5)

        # dense layer + output
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
