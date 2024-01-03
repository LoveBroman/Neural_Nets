import numpy as np
from layers import *

class NeuralNet:
     def __init__(self, layers=None):
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


     def back_prop(self, batch, y):
         w_delt = []
         deltas = [] * len(self.layers)
         for i, l in enumerate(self.layers):
             deltas[i] = np.zeros(l.W.shape)
         os = []
         for x, i in enumerate(batch):
             self.for_prop(x)
             for j in reversed(range(len(self.layers)))[:-1]:
                 if j == len(self.layers) -1:
                     eo = y[i] - self.layers[j].out
                     deltas[j] = self.layers[j - 1].out * eo
                 else:
                     dact = np.vectorize(derivate(self.layers[j].act))
                     eo = dact(self.layers[j].out)
                     dw = self.layers[j -1] @ eo.reshape(1, len(eo))
                     deltas[j] = dw
         for i, d in enumerate(deltas):
             self.layers[i] -= self.a * d

     def fit(self, X, y):
        for i in range(len(X)):
            yhat = self.for_prop(X[i])
            self.back_prop(yhat, y)


