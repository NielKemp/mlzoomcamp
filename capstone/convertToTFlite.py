#importing required packages

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.lite as tflite

#load current model in h5 format
model = keras.models.load_model('xception_fin_23_0.880.h5')

#initiate converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#convert model to tflite format
tflite_model = converter.convert()
#save tflite model
with open('xception_fin_23_0_880.tflite', 'wb') as _out_:
    _out_.write(tflite_model)
