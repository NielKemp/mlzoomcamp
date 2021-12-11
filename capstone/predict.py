#import packages
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

#load model in h5 format
model = keras.models.load_model('xception_fin_23_0.880.h5')

img = np.array([np.array(load_img('rose_test.jpg', target_size = (299,299)))])

proc_img = preprocess_input(img)

prediction = model.predict(proc_img)

classes = ['daisy','dandelion','rose','sunflower','tulip']
print(dict(zip(classes, prediction.round(3)[0])))
