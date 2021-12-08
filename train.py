import pandas as pd
import numpy as np
import os

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
import logging
from models.CustomModel import CustomModel
from models.ImageGenerator import ImageGenerator
from tensorflow.keras.utils import plot_model

from utils.helpers import load_utkface_npz, mk_dir, split_data



logging.basicConfig(level=logging.DEBUG)

BATCH_SIZE = 64
NUM_EPOCHS = 100
DATASET = "UTKFace"
MODEL_NAME = "CUSTOM_3"
DATA_DIR = f"./datasets/{DATASET}"



SAVE_DIR = os.path.abspath(f"./saves")
# SAVE_DIR = os.path.abspath(f"./saves/{DATASET}/{MODEL_NAME}")

def main():

    logging.info("Loading Data.... ")

    filenames, age_classes, sex_classes, _, img_size = load_utkface_npz("./datasets/utkface.npz")

    ground_truth = dict(zip(filenames, zip(sex_classes, age_classes)))

    train_keys, validation_keys = split_data(ground_truth, do_shuffle=True)

    img_generator = ImageGenerator(
        ground_truth=ground_truth,
        batch_size=BATCH_SIZE,
        image_size=int(img_size),
        train_keys=train_keys,
        validation_keys=validation_keys,
        path_prefix=os.path.abspath(DATA_DIR)
    )

    model = CustomModel(image_size=img_size, image_shape=(img_size, img_size, 3))()

    optimizer = SGD(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=["binary_crossentropy",
              "categorical_crossentropy"],
        metrics={'age': 'accuracy', 'sex': 'accuracy'}
    )
    
    logging.info("Model summary...")
    model.count_params()
    model.summary()

    logging.info("Saving model...")

    save_dir = f"{SAVE_DIR}/{DATASET}"
    mk_dir(save_dir)
    save_dir = f"{save_dir}/{MODEL_NAME}"
    mk_dir(save_dir)
    checkpoint_dir = f"{save_dir}/checkpoint"
    mk_dir(checkpoint_dir)
    plot_model(model, to_file=f"{save_dir}/{MODEL_NAME}.png")

    with open(os.path.join(f"{save_dir}/{MODEL_NAME}.json"), "w") as f:
        f.write(model.to_json())

    reduce_lr = ReduceLROnPlateau(
        verbose=1, min_delta=0.001, patience=4)

    callbacks = [
        reduce_lr,
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto"
        ),
    ]

    logging.info("Running training...")

    hist = model.fit(
        img_generator.flow(mode="train"),
        steps_per_epoch=int(len(train_keys) / BATCH_SIZE),
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=img_generator.flow(mode='val'),
        validation_steps=int(len(validation_keys) / BATCH_SIZE)
    )

    logging.debug("Saving weights...")
    model.save_weights(os.path.join(f"{save_dir}/{MODEL_NAME}.h5"), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(save_dir, 'history_'+MODEL_NAME+'.h5'), "history")









if __name__ == "__main__":
    main()