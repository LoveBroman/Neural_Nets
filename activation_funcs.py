import numpy as np

def relu(a):
    return max(0,a)

def sigmoid(a):
    return 1/(1+np.exp(-a))

def mse(y, yhat):
    return sum ((y - yhat) ** 2)