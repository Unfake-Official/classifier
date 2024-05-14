import tensorflow as tf

# CNN model sketch:


class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()

        # convolution + feature extraction
        # todo: change filters size
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3))
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.Activation('relu')
        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        
        self.dropout = tf.keras.layers.Dropout(0.5)

        # dense layer + output
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, x):
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
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
