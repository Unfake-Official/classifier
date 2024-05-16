import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=False):
        identity = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_planes = 16
        self.embDim = 128 * block.expansion
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(planes, stride))
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def get_embedding_dim(self):
        return self.embDim