    keras.layers.Embedding(88000, 200),
    keras.layers.LSTM(200, return_sequences=True, input_shape=(None, 1)),
    keras.layers.Dense(1, activation="sigmoid")