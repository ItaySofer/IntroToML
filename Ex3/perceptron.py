import numpy as np


class Perceptron:

    def __init__(self, dim):
        self.w = [0] * dim

    def train(self, x, y):
        assert(len(x) == len(self.w))

        for xt, yt in zip(x, y):
            predict = np.sign(np.dot(x, self.w))
            if predict != yt:
                self.w += yt*xt

    def predict(self):
        pass