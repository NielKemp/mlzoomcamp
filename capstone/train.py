import numpy as np
#import pandas as pd



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=40,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4,
                                   shear_range=0.2,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split = 0.2)


train_generator = train_datagen.flow_from_directory(
    'flowers/',
    target_size=(299, 299), #need to use 299 here because of the requirements of the Xception model
    batch_size=32,
    class_mode = 'categorical',
    subset='training', shuffle = True, seed = 69420) # set as training data

valid_generator = train_datagen.flow_from_directory(
    'flowers/',
    target_size=(299, 299), #need to use 299 here because of the requirements of the Xception model
    batch_size=32,
    class_mode = 'categorical',
    subset='validation', shuffle = True, seed = 69420) # set as validation data

base_model = Xception(weights = 'imagenet', include_top = False, input_shape = (299,299,3))

base_model.trainable = False

inputs = keras.Input(shape=(299,299,3))
base = base_model(inputs, training = False)
vectors = keras.layers.GlobalAveragePooling2D()(base)

hidden1 = keras.layers.Dense(512, activation = 'relu')(vectors)
hidden2 = keras.layers.Dense(128, activation = 'relu')(hidden1)
hidden3 = keras.layers.Dense(32, activation = 'relu')(hidden2)

outputs = keras.layers.Dense(5, activation = 'softmax')(hidden3)

finalModel = keras.Model(inputs, outputs)
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.75)
loss = keras.losses.CategoricalCrossentropy(from_logits=False)

finalModel.compile(optimizer = optimizer, loss=loss, metrics=['accuracy'])

print(finalModel.summary())

#checkpoint = keras.callbacks.ModelCheckpoint(
#    'xception_fin_{epoch:02d}_{val_accuracy:.3f}.h5',
#    save_best_only=True,
#    monitor='val_accuracy',
#    mode='max'
#)

history = finalModel.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=1,
    validation_data=valid_generator,
    validation_steps=20#,
#    callbacks=[checkpoint]

)
