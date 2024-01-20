import numpy as np
from utils import *
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, input):
        pass

class Dense(Layer):
    def __init__(self, insize, outsize, act="linear"):
        self.insize = insize
        #self.input = input
        self.W = np.random.rand(self.insize, outsize).T
        self.bias = np.random.rand()
        self.act = act

    def forward(self, input):
        if isinstance(input, float) or isinstance(input, int):
            a = input * self.W
        else:
            a = self.W @ input
        if self.act == "linear":
            self.out = a
        elif self.act == "sigmoid":
            self.out = sigmoid(a)
        elif self.act == "tanh":
            self.out = np.tanh(a)
        elif self.act == "relu":
            self.out = relu(a)
        return self

class Input(Layer):
    def __init__(self, insize):
        self.insize = insize

    def forward(self, input):
        self.out = input
        return self