import os
import numpy as np
from numpy.random import shuffle
from tensorflow.keras.utils import to_categorical
import cv2
from tensorflow.keras.preprocessing.image import random_rotation, random_shear, random_shift, random_zoom
from utils.utkface_preprocess import get_age_class

class ImageGenerator(object):

    def __init__(self, ground_truth, batch_size: int, image_size: int, train_keys, validation_keys, path_prefix:str):
        self._ground_truth = ground_truth
        self._batch_size = batch_size
        self._image_size = image_size
        self._train_keys = train_keys
        self._train_keys = train_keys
        self._validation_keys = validation_keys
        self._path_prefix = path_prefix

    def augment_data(self, image_array):

        image_array = image_array[:,::-1]
        """
        if np.random.random() > 0.5:
            image_array = random_crop(image_array,4)
        """
        if np.random.random() > 0.75:
            image_array = random_rotation(image_array, 20, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            image_array = random_shear(image_array, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            image_array = random_shift(image_array, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            image_array = random_zoom(image_array, [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
            
        return image_array

    def flow(self, mode="train"): 
        while True:
            if mode == 'train':
                shuffle(self._train_keys)
                keys = self._train_keys
            elif mode == 'val' or mode == 'demo':
                shuffle(self._validation_keys)
                keys = self._validation_keys
            else:
                raise Exception('invalid mode: %s' % mode)


            inputs, targets = [], []
            for key in keys:
                image_path = os.path.join(self._path_prefix, key)
                image_array = cv2.imread(image_path)
                image_array = cv2.resize(src=image_array, dsize=(self._image_size, self._image_size))

                num_image_channels = len(image_array.shape)
                if num_image_channels != 3:
                    continue

                ground_truth = self._ground_truth[key]

                image_array = image_array.astype('float32')

                if mode == "train":
                    image_array = self.augment_data(image_array)

                inputs.append(image_array)
                targets.append(ground_truth)
                if len(targets) == self._batch_size:
                    inputs = np.asarray(inputs)
                    targets = np.asarray(targets)

                    sex = to_categorical(targets[:, 0], 2)
                    age = list(map(get_age_class, targets[:, 1]))
                    age = to_categorical(age, 11)

                    yield self._wrap_in_dictionary(inputs, sex, age)
                    inputs, targets, sex, age = [],  [], [], []


    def _wrap_in_dictionary(self, image_array, sex, age_quantized):
        return [
            {'input_1': image_array},
            {'sex': sex, 'age': age_quantized}
        ]



                


            






