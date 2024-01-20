import numpy as np
from layers import *
from utils import *

class NeuralNet:
    def __init__(self, alpha, epochs, batch_size=None, layers=None):
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []
    def add_layer(self, layer):
        self.layers.append(layer)
    def for_prop(self, input):
        nextout = input
        for layer in self.layers:
            nextout = layer.forward(nextout).out
        return nextout

    def predict(self, X):
        pred = np.vectorize(self.for_prop)
        return pred(X)

    def back_prop(self, batch, y):
        deltas = [None] * (len(self.layers) - 1)
        w_deltas = [None] * (len(self.layers) - 1)
        for i in range(len(self.layers))[:-1]:
            deltas[i] = np.zeros(self.layers[i+1].W.shape[0])
            w_deltas[i] = np.zeros(self.layers[i+1].W.shape)
        for i, x in enumerate(batch):
            self.for_prop(x)
            for j in list(reversed(range(len(self.layers))))[:-1]:
                if j == len(self.layers) - 1:
                    eo = y[i] - self.layers[j].out
                    dact = np.vectorize(derivate(self.layers[j].act))
                    deltas[j-1] = dact(self.layers[j].out) * eo
                else:
                    dact = np.vectorize(derivate(self.layers[j].act))
                    eo = dact(self.layers[j].out)
                    deltas[j-1] = eo * np.matmul(self.layers[j].W, deltas[j].T)

            for j in range(len(deltas)):
                if isinstance(self.layers[j].out, int) or isinstance(self.layers[j].out, np.float64):
                    w_deltas[j] += deltas[j].reshape(len(deltas[j]), 1) * self.layers[j].out
                else:
                    if j == 2:
                        pass
                    print(f"j is {j}")
                    w_deltas[j] += deltas[j] @ self.layers[j].out.T
        for i in range(len(self.layers))[1:]:
            if i == 2:
                pass
            self.layers[i].W += self.alpha * w_deltas[i - 1]


    def get_batch(self, X, y):
        rg = RandGen(len(y))
        binds = rg.getk(self.batch_size)
        xbatch, ybatch = X[binds], y[binds]
        X, y = np.delete(X, binds, 0), np.delete(y, binds, 0)
        return xbatch, ybatch, X, y

    def fit(self, X, y):
        for i in range(self.epochs):
            if self.batch_size is None:
                self.back_prop(X, y)
            else:
                while len(y) != 0:
                    Xbatch, ybatch, X, y = self.get_batch(X, y)
                    self.back_prop(Xbatch, ybatch)