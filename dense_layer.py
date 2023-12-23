import numpy as np
from activation_funcs import *

class Dense:
    def __init__(self, input, outsize, act="linear"):
        self.insize = len(input)
        self.input = input
        self.W = np.random.rand(self.insize, outsize)
        self.act = act

    def forward(self):
        a = np.matmul(self.W, self.input)
        if self.act == "linear":
            return a
        elif self.act == "sigmoid":
            return sigmoid(a)
        elif self.act == "tanh":
            return np.tanh(a)
        elif self.act == "relu":
            return relu(a)


