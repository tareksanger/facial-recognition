import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras import backend as K

from keras.applications.mobilenet import MobileNet

class MobileNetReg:
    def __init__(self, image_size, alpha):
        

        logging.info(f"Image Data Format: {K.image_data_format()}")
        if K.image_data_format() == 'channels_first':
            self._input_shape = (3, image_size, image_size)
        else:
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.alpha = alpha

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_mobilenet = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling=None)
        x = model_mobilenet(inputs)
        #flatten = Flatten()(x)
        
        feat_a = Conv2D(20,(1,1),activation='relu')(x)
        feat_a = Flatten()(feat_a)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu',name='feat_a')(feat_a)

        pred_a = Dense(1,name='pred_a')(feat_a)
        model = Model(inputs=inputs, outputs=[pred_a])


        return model
