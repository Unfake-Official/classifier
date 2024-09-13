from keras import layers, Model


class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()

        self.lstm1 = layers.LSTM(1024, input_shape = None)
        self.drop1 = layers.Dropout(0.2)
        self.lstm2 = layers.LSTM(512)
        self.drop2 = layers.Dropout(0.2)
        self.lstm3 = layers.LSTM(256)
        self.drop3 = layers.Dropout(0.2)
        self.lstm4 = layers.LSTM(128)
        self.drop4 = layers.Dropout(0.2)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(16, activation='relu')
        self.dense4 = layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.lstm1(x)
        x = self.drop1(x)
        x = self.lstm2(x)
        x = self.drop2(x)
        x = self.lstm3(x)
        x = self.drop3(x)
        x = self.lstm4(x)
        x = self.drop4(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)

        return x
