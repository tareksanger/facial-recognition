from scipy.io import loadmat
from datetime import datetime
import os
import numpy as np
from numpy.random import shuffle

def calc_age(date_taken: int, date_of_birth): 
    """
        Calcualtes the age based on the date of birth and the date the photo was taken.
    """
    birth = datetime.fromordinal(max(int(date_of_birth) - 366, 1))
    # Balance out the age based on the time of year
    return date_taken - birth.year if birth.month < 7 else date_taken - birth.year - 1


def get_meta(mat_path: str, db: str):
    """
        Loads the Meta Data provided from the IMDB files.
    """
    meta = loadmat(mat_path)
    full_path: list[str] = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender: list[int] = meta[db][0, 0]["gender"][0]
    photo_taken: list[int] = meta[db][0, 0]["photo_taken"][0]  # year
    face_score: list[int] = meta[db][0, 0]["face_score"][0]
    second_face_score: list[int] = meta[db][0, 0]["second_face_score"][0]
    age: list[int] = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path: str):
    d = loadmat(mat_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]

def load_data_npz(npz_path: str):
    """
        Loads the saved NPZ file
    """
    d = np.load(npz_path)

    return d["image"], d["gender"], d["age"], d["img_size"]


def load_utkface_npz(npz_path: str):
    
    d = np.load(npz_path)

    return d["filenames"], d["age_classes"], d["sex_classes"], d['race_classes'], d['img_size']


def split_data(data, validation_split=0.2, do_shuffle=False):
    data_keys = sorted(data.keys())

    if do_shuffle:
        shuffle(data_keys)

    training_split = 1 - validation_split
    num_train = int(training_split * len(data_keys))
    train_keys = data_keys[:num_train]
    validation_keys = data_keys[num_train:]
    return train_keys, validation_keys

def mk_dir(dir):
    """
        Creates a File Directory
    """
    try:
        os.mkdir( dir )
    except OSError:
        pass
