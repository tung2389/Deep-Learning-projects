from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob
import shutil

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

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


def caculateNumOfFiles():
  total_train, total_val = 0, 0
  for path, subdirs, files in os.walk(train_dir):
    for name in files:
      total_train += 1
  
  for path, subdirs, files in os.walk(val_dir):
    for name in files:
      total_val += 1
  
  return [total_train, total_val]

def returnImportantInfo():
    organizeAllImagesToFolders()
    total_train, total_val = caculateNumOfFiles()
    return [train_dir, val_dir, total_train, total_val]

def prepareTraingAndValSet(BATCH_SIZE, IMG_SHAPE):
    train_dir, val_dir, total_train, total_val = returnImportantInfo()
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
    return [train_data_gen, val_data_gen]