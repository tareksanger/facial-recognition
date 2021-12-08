import logging
import sys
from keras.layers.normalization.batch_normalization import BatchNormalization
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras import backend as K

from keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling2D, MaxPooling2D



class CustomModel(object):

    def __init__(self, image_size: int, image_shape):
        self._image_size = image_size
        self._image_shape = image_shape
        pass
    

    def __call__(self):
        inputs = Input(shape=self._image_shape)

        x = Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = Conv2D(filters=32*2, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(filters=32*3, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(filters=32*4, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(filters=32*5, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(filters=32*6, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        bottleneck = GlobalMaxPooling2D()(x)

        # For sex calculation
        x = Dense(units=128, activation='relu')(bottleneck)
        sex_output = Dense(units=2, activation='softmax', name="sex")(x)

        # For Age Calculation
        x = Dense(units=128, activation='relu')(bottleneck)
        age_output = Dense(units=11, activation='softmax', name="age")(x)

        model = Model(inputs=inputs, outputs=[sex_output, age_output])

        return model



















        






