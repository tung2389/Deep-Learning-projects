import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from utils.prepareReviewDataset import intToWord, return_processed_data_and_labels

def decode_review(text):
	return " ".join([intToWord.get(i, "?") for i in text])

train_data, train_labels, test_data, test_labels = return_processed_data_and_labels(450)
# train_data = train_data[..., np.newaxis] # Add one more dimension to train data. Now it has shape (25000, 250, 1)
# test_data = test_data[..., np.newaxis]
model = keras.models.Sequential([
    keras.layers.Embedding(88000, 200),
    keras.layers.LSTM(200, input_shape=(None, 1)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

x_train = train_data[10000:]
y_train = train_labels[10000:]

x_val = train_data[:10000]
y_val = train_labels[:10000]

model_checkpoint = keras.callbacks.ModelCheckpoint(
    os.getcwd() + "/Text Classification/LSTM model.h5", 
    save_best_only=True)
    # monitor="val_acc" ,
    # mode="max")

model.fit(x_train, y_train, 
            epochs=60, 
            batch_size=512,
            validation_data=(x_val, y_val), callbacks=[model_checkpoint])