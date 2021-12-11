#packages
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

#create preprocessor object for xception model
preprocessor = create_preprocessor('xception', target_size = (299,299))

#import model
interpreter = tflite.Interpreter(model_path = 'xception_fin_23_0_880.tflite')
interpreter.allocate_tensors()

#get input/output indexes
in_index = interpreter.get_input_details()[0]['index']
out_index = interpreter.get_output_details()[0]['index']

#initate classes into list 
flower_classes = ['daisy','dandelion','rose','sunflower','tulip']


def predict(url):
    image_dat = preprocessor.from_url(url)
    
    interpreter.set_tensor(in_index, image_dat)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(out_index).round(5)[0].tolist()
    
    return dict(zip(flower_classes, predictions))
    
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
    
    

