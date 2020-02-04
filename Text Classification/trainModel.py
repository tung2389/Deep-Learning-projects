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

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

def saveTheModel():
	model.save("model.h5")
    
def printModelEvaluation():
	results = model.evaluate(test_data, test_labels)
	print(results)

def testModelOnTestData():
	test_review = test_data[0]
	predict = model.predict([test_review])
	print("Review: ")
	print(decode_review(test_review))
	print("Prediction: " + str(predict[0]))
	print("Actual: " + str(test_labels[0]))

