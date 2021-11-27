#import packages
import tflite_runtime.interpreter as tflite
from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np

#import model
interpreter = tflite.Interpreter(model_path = 'cats-dogs-v2.tflite')
interpreter.allocate_tensors()


#get input and output index
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


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
    
    
def predict(url):
    img_ = download_image('https://upload.wikimedia.org/wikipedia/commons/1/18/Vombatus_ursinus_-Maria_Island_National_Park.jpg')    
    img_prep = prepare_image(img_, target_size = (150,150))

    x = np.array(img_prep)
    X = np.array([x],dtype=np.float32)
    X = X * 1./255
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index).tolist()
    
    return(preds)
    
    
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
