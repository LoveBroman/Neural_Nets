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
        self.W = np.random.rand(self.insize, outsize)
        self.bias = np.random.rand()
        self.act = act

    def forward(self, input):
        a = np.matmul(self.W, input)
        if self.act == "linear":
            self.out = a
        elif self.act == "sigmoid":
            self.out = sigmoid(a)
        elif self.act == "tanh":
            self.out =  np.tanh(a)
        elif self.act == "relu":
            self.out = relu(a)



