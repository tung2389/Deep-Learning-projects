from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

test_images = test_images/255.0
preprocessed_test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

model = keras.models.load_model(cwd + "/Fashion MNIST/model.h5")
predictions = model.predict(preprocessed_test_images)

def test_the_model():
    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[test_labels[i]])
        plt.title(class_names[np.argmax(predictions[i])])
        plt.show()
        
def print_test_acc():
    test_loss, test_acc = model.evaluate(preprocessed_test_images, test_labels, batch_size=128)
    print("Tested Accurary: ", test_acc)

print_test_acc()