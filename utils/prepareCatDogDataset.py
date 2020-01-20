from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def prepareDataset():
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

    zip_dir = keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

    base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val
    return [train_dir, validation_dir, total_train, total_val]

def prepareTraingAndValSet(BATCH_SIZE, IMG_SHAPE):
    train_dir, validation_dir, total_train, total_val = prepareDataset()
    image_gen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
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
                                                    class_mode='binary')

    #classes based on subdirectory under training_dir, in this case cats and dogs

    #train_data_gen shape: [x,y]; x shape: number_of_batch -> batch_size -> IMG_SHAPE -> IMG_SHAPE -> channels (in this case 3)
    #train_data_gen shape: [[[[[R,G,B]]]],[labels]]
    # augmented_images = [train_data_gen[0][0][0] for i in range(5)] # take 5 random images in the first batch
    # plotImages(augmented_images)

    image_gen_val = ImageDataGenerator(rescale=1./255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')
    
    return [train_data_gen, val_data_gen]