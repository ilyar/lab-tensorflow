from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class Mnist(Model):
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
