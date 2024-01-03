import numpy as np

def relu(a):
    return max(0,a)

def sigmoid(a):
    return 1/(1+np.exp(-a))

def mse(y, yhat):
    return sum ((y - yhat) ** 2)

def derivate(fs):
    if fs == "relu":
        return lambda x: int(x > 0)
    elif fs == "sigmoid":
        return lambda x: sigmoid(x) (1 - sigmoid(x))
    else:
        return lambda x: (fs(x + 1e-7) - fs(x))/ 1e-7


#Generates randomnumbers a different one each time.
class RandomGen:
    def __init__(self, n):
        self.n = n
        self.nums = []

    def next(self):
        num = np.random.randint(self.n)
        while num in self.nums:
            num = np.random.randint(self.n)
        self.nums.append(num)
        return num

    def getk(self, k):
        for i in range(k):
            self.next()
        return self.nums



