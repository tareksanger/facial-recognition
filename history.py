import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Plots training curves from history file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to history h5 file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input

    df = pd.read_hdf(input_path, "history")
    print(df.columns)
    input_dir = os.path.dirname(input_path)
    plt.plot(df["sex_loss"], label="loss (sex)")
    plt.plot(df["age_loss"], label="loss (age)")
    plt.plot(df["val_sex_loss"], label="val_loss (sex)")
    plt.plot(df["val_age_loss"], label="val_loss (age)")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "loss.png"))
    plt.cla()

    plt.plot(df["sex_accuracy"], label="accuracy (sex)")
    plt.plot(df["age_accuracy"], label="accuracy (age)")
    plt.plot(df["val_sex_accuracy"], label="val_accuracy (sex)")
    plt.plot(df["val_age_accuracy"], label="val_accuracy (age)")
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(input_dir, "accuracy.png"))


if __name__ == '__main__':
    main()