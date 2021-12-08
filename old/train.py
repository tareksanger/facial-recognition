import pandas as pd
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from keras.utils import np_utils
from utils.callbacks import DecayLearningRate
from utils.generators import *
from utils.helpers import mk_dir, load_data_npz
import sys
import numpy as np
from keras.applications.mobilenet import MobileNet
# import TYY_callbacks
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from moviepy.editor import *
import cv2
from models.MobileNetEstimator import MobileNetEstimator

logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--input", 
        "-i", 
        type=str, 
        required=True,
        help="path to input database npz file")

    parser.add_argument(
        "--db", 
        type=str, 
        required=True,
        default="imdb",
        help="database name")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input
    db_name = args.db
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split

    logging.debug("Loading data...")
    images, gender, age, image_size = load_data_npz(input_path)

    x_data = images
    y_data_a = age
    y_data_g = gender

    # y_data = np.array(list(zip(y_data_g, y_data_a)))

    # Why SDG?
    opt = SGD(learning_rate=0.001)

    # Why alpha 0.25? What does it do?
    alpha = 0.25 #1

    model = MobileNetEstimator(
        image_size=image_size, 
        alpha=alpha,
        num_neu=21,
        weights='imagenet'
    )()

    save_name = "MobileNet_"
    model.compile(
        optimizer=opt,
        loss=["binary_crossentropy",
              "categorical_crossentropy"],
        metrics={'gender': 'accuracy',
                 'age': 'accuracy'},
    )
    
    start_decay_epoch = [30,60]

    logging.info("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir(db_name+"_models")
    mk_dir(db_name+"_models/"+save_name)
    mk_dir(db_name+"_checkpoints")
    plot_model(model, to_file=db_name+"_models/"+save_name+"/"+save_name+".png")

    with open(os.path.join(db_name+"_models/"+save_name, save_name+'.json'), "w") as f:
        f.write(model.to_json())

    decaylearningrate = DecayLearningRate(start_decay_epoch)

    callbacks = [
        ModelCheckpoint(db_name+"_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate
                        ]

    logging.debug("Running training...")
    
    data_num = len(x_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    x_data = x_data[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    
    x_train = x_data[:train_num]
    x_test = x_data[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    hist = model.fit(data_generator_reg(X=x_train, Y=y_train_a, batch_size=batch_size),
                               steps_per_epoch=train_num // batch_size,
                               validation_data=(x_test, [y_test_a]),
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)
    
    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models/"+save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models/"+save_name, 'history_'+save_name+'.h5'), "history")






if __name__ == "__main__":
    main()