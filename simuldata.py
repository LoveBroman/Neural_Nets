import numpy as np
import matplotlib.pyplot as plt

def cosy_noise(n, dev):
    data = np.linspace(0, 10, n)
    noise = dev * np.random.randn(n)
    y = np.cos(data) + noise
    return data, y

