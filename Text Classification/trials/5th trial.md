model = keras.models.Sequential([
    keras.layers.Embedding(88000, 128),
    keras.layers.LSTM(80),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

batch_size = 1000
epochs = 15
maxlen = 250