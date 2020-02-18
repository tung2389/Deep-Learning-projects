import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from utils.prepareReviewDataset import intToWord, return_processed_data_and_labels

def decode_review(text):
	return " ".join([intToWord.get(i, "?") for i in text])

train_data, train_labels, test_data, test_labels = return_processed_data_and_labels(250)
# train_data = train_data[..., np.newaxis] # Add one more dimension to train data. Now it has shape (25000, 250, 1)
# test_data = test_data[..., np.newaxis]
model = keras.models.Sequential([
    keras.layers.Embedding(88000, 80),
    keras.layers.LSTM(60),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

x_train = train_data[5000:]
y_train = train_labels[5000:]

x_val = train_data[:5000]
y_val = train_labels[:5000]

model_checkpoint = keras.callbacks.ModelCheckpoint(
    os.getcwd() + "/Text Classification/LSTM model.h5", 
    save_best_only=True,
    monitor="val_acc",
    mode="max")

model.fit(x_train, y_train, 
            epochs=15, 
            batch_size=1000,
            validation_data=(x_val, y_val), callbacks=[model_checkpoint])

def evaluateModel():
    model = keras.models.load_model(os.getcwd() + '/Text Classification/LSTM model.h5')
    print(model.evaluate(test_data, test_labels))

evaluateModel()

def analyzeData():
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    data = keras.datasets.imdb
    (raw_train_data, raw_train_labels), (raw_test_data, raw_test_labels) = data.load_data(num_words = 88000)
    for review in raw_train_data:
        if len(review) <= 100:
            count1 = count1 + 1
        elif 100 < len(review) <= 200:
            count2 = count2 + 1
        elif 200 < len(review) <= 300:
            count3 = count3 + 1
        elif len(review) > 300:
            count4 = count4 + 1
    
    print("Reviews whose lengths are smaller than 100: " , count1)
    print("Reviews whose lengths are between 100 and 200: " , count2)
    print("Reviews whose lengths are between 200 and 300: " , count3)
    print("Reviews whose lengths are bigger than 400: " , count4)

