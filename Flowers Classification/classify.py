import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np 
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from utils.prepareGPU import prepareGPU
from utils.prepareFlowersDataset import returnImportantInfo, prepareTraingAndValSet

session = prepareGPU()

train_dir, val_dir, total_train, total_val = returnImportantInfo()

BATCH_SIZE = 100
IMG_SHAPE = 150

#Image augmentation
train_data_gen, val_data_gen = prepareTraingAndValSet(BATCH_SIZE, IMG_SHAPE)

model = keras.models.Sequential([
  keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Dropout(0.5),
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 60
history = model.fit_generator(
  train_data_gen,
  steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
  epochs=epochs,
  validation_data=val_data_gen,
  validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

cwd = os.getcwd()
model.save(cwd + "/Flowers Classification/model.h5")

def plotHistoryGraph():
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

plotHistoryGraph()