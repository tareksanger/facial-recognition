import os
from glob import glob
import pandas as pd
import numpy as np
from utils.utkface_preprocess import parse_utk_meta


DATA_DIR = os.path.abspath("./datasets/UTKFace")
TRAIN_TEST_SPLIT = 0.8
IM_WIDTH = IM_HEIGHT = IMG_SIZE = 198
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())


def main():

    # Extract details from the files
    filename_list, age_classes, sex_classes, race_classes = parse_utk_meta(DATA_DIR)

    attributes = list(zip(filename_list, age_classes, sex_classes, race_classes))

    df = pd.DataFrame(attributes)
    df.columns = ['filename','age', 'sex', 'race']
    df = df.dropna()

    np.savez(
        "./datasets/utkface.npz", 
        filenames=df["filename"].to_list(), 
        age_classes=df["age"].to_list(), 
        sex_classes=df["sex"].to_list(), 
        race_classes=df["race"].to_list(), 
        img_size=IMG_SIZE)



if __name__ == "__main__":
    main()