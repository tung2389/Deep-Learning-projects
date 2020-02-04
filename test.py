import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

np.random.seed(0) 

SLENG = 20 # sequence length
# numpy array
seq = np.arange(0, SLENG*SLENG, SLENG)
print(seq)
# 0  20  40  60  80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380

# model needs X as input and y as ouptut shapes
X = seq.reshape(1, SLENG, 1)
y = seq.reshape(1, SLENG)

# define LSTM configuration
n_neurons = SLENG
n_batch = 1
n_epoch = 1500

# create LSTM net
model = Sequential()
model.add(LSTM(n_neurons, activation="relu", input_shape=(SLENG, 1)))
model.add(Dense(SLENG))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print(model.summary())