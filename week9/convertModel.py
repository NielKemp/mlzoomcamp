#import packages
import numpy as np
import tensorflow as tf
from tensorflow import keras

#load model in h5 format
model = keras.models.load_model('dogs_cats_10_0.687.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('dogs_cats_10_0.687.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
    
#import tensofrlow lite package
import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path = 'dogs_cats_10_0.687.tflite')
interpreter.allocate_tensors()


#get input and output index
input_index = interpreter.get_input_details()[0]['index']
print(f'input index: {input_index}')

output_index = interpreter.get_output_details()[0]['index']
print(f'output index: {output_index}')



#use provided image
from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
    
    
img_ = download_image('https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg')    
img_prep = prepare_image(img_, target_size = (150,150))

x = np.array(img_prep)
X = np.array([x],dtype=np.float32)
X = X * 1./255

#print(X)

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
print(f'prediction: {preds}')
