import logging

import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D

class MobileNetEstimator(object):
    """
    
    """

    def __init__(self, image_size, alpha, num_neu, weights=None):
        logging.info(f"Image Data Format: {K.image_data_format()}")
        if K.image_data_format() == 'channels_first':
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        
        self.alpha = alpha
        self.num_neu = num_neu
        self.weights = weights
        self.FC_LAYER_SIZE = 1024

    def __call__(self):
        logging.info("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_mobilenet = MobileNet(
            input_shape=self._input_shape, 
            alpha=self.alpha, 
            depth_multiplier=1, 
            dropout=1e-3,
            include_top=False, 
            weights=self.weights, 
            input_tensor=None, 
            pooling=None)

        x = model_mobilenet(inputs)

        feat_a = GlobalAveragePooling2D()(x)
        feat_a = Dropout(0.5)(feat_a)
        feat_a = Dense(self.FC_LAYER_SIZE, activation="relu")(feat_a)

        pred_g_softmax = Dense(2, activation='softmax', name='sex')(feat_a)
        pred_a_softmax = Dense(self.num_neu, activation='softmax', name='age')(feat_a)

        model = Model(inputs=inputs, outputs=[pred_g_softmax, pred_a_softmax])

        return model
