import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
import math

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_images)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), padding='same', activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10,  activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

NO_OF_EXAMPLES = len(train_images)
BATCH_SIZE = 32

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
# shape[0] is 60000; Each image has 28 rows and 28 cols, each rows has 28 element; Each element
# is an array which have only one child. If the image has multiple channels, for example 3 channels,
# each element in each row will have 3 children
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

model.fit(
    train_images,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=10,
    shuffle=True,
    steps_per_epoch=math.ceil(NO_OF_EXAMPLES/BATCH_SIZE)
)

model.save("model.h5")