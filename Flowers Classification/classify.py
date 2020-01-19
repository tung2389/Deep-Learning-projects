import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import glob
import shutil
import numpy as np 
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from utils.prepareGPU import prepareGPU

session = prepareGPU()

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

def organizeAllImagesToFolders():
  for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg') #Find all the image file belong to a specific class
    print("{}: {} Images".format(cl, len(images)))  #Print classname and number of images belong to that class
    train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):] #train test takes 80 percent, validation test takes 20 percent

    for t in train:
      if not os.path.exists(os.path.join(base_dir, 'train', cl)):
        os.makedirs(os.path.join(base_dir, 'train', cl))
      shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
      if not os.path.exists(os.path.join(base_dir, 'val', cl)):
        os.makedirs(os.path.join(base_dir, 'val', cl))
      shutil.move(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

def caculateNumOfFiles():
  total_train, total_val = 0, 0
  for path, subdirs, files in os.walk(train_dir):
    for name in files:
      total_train += 1
  
  for path, subdirs, files in os.walk(val_dir):
    for name in files:
      total_val += 1
  
  return [total_train, total_val]


total_train, total_val = caculateNumOfFiles()[0], caculateNumOfFiles()[1]

BATCH_SIZE = 100
IMG_SHAPE = 150

#Image augmentation
image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=90,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2, #fix one axis and stretch the image at a certain angle known as the shear angle
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='sparse')

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')

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