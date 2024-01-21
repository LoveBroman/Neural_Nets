from layers import Dense, Input
from network import NeuralNet
from simuldata import *
# from tensorflow import keras
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Activation

# kmod = Sequential()
# kmod.add(Dense(3, input_shape=(1,), activation="relu"))
# kmod.add(Dense(1, activation="sigmoid"))

X, y = cosy_noise(100, 0)
model = NeuralNet(.001, 100)
model.add_layer(Input(1))
model.add_layer(Dense(1, 6, act="tanh"))
#model.add_layer(Dense(2, 2, act="relu"))
model.add_layer(Dense(6, 1, act="linear"))


model.fit(X, y)
preds = model.predict(X)
print(preds)
plt.plot(X, preds)
plt.plot(X, y, "r")
plt.show()