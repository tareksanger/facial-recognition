
# LEARNING_RATE = 1e-3
# BATCH_SIZE = 32
# IMG_DIMS = (224, 224, 3)
# NUM_EPOCHS = 70
# FINAL_WEIGHTS_PATH = 'final_weights.hdf5'
import os

DATA_DIR = os.path.abspath("../assets/datasets/UTKFace")
TRAIN_TEST_SPLIT = 0.8
IM_WIDTH = IM_HEIGHT = 198
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())