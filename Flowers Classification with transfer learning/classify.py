import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from utils.prepareGPU import prepareGPU
from utils.prepareFlowersDataset import returnImportantInfo, prepareTraingAndValSet
train_dir, val_dir, total_train, total_val = returnImportantInfo()

CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224
BATCH_SIZE = 32

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
all_layers_except_the_last = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))

all_layers_except_the_last.trainable = False

model = keras.Sequential([
    all_layers_except_the_last,
    keras.layers.Dense(5, activation="softmax")
])

model.summary()

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

train_data_gen, val_data_gen = prepareTraingAndValSet(BATCH_SIZE, IMAGE_RES)

EPOCHS = 6
with tf.device('/CPU:0'): #Cannot use GPU because the neural network is so big, my GPU memory is not big enough
  history = model.fit_generator(
      train_data_gen,
      steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
      epochs=EPOCHS,
      validation_data=val_data_gen,
      validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
  )

model.save(os.getcwd() + "/Flowers Classification with transfer learning/model.h5")