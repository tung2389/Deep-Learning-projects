    keras.layers.Embedding(88000, 500),
    keras.layers.LSTM(500, input_shape=(None, 1)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")