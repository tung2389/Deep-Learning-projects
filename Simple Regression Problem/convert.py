import tensorflow as tf 
from tensorflow import keras
import numpy as np 

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38, 120, 60, 80, 90, 40],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100, 248, 140, 176, 194, 104],  dtype=float)

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1]) 
    #Have input layer with 1 input and output layer with 1 output   
])

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(1.5))
model.fit(celsius_q, fahrenheit_a, epochs = 100, verbose=1)
print(model.predict([100]))
