import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils.helpers import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        required=True,
        help="path to output database mat file")
    
    parser.add_argument(
        "--db", 
        type=str,
        default="imdb",
        help="dataset; wiki or imdb")
    
    parser.add_argument(
        "--img_size", 
        '-s', 
        type=int, 
        default=64,
        help="output image size")

    parser.add_argument(
        "--score_threshold", 
        '-t', 
        type=float, 
        default=1.0,
        help="minimum face_score")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    # Get root path and the mat file path.
    root_path = "./datasets/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)

    # Read the Meta data
    full_path, _, gender, _, face_score, second_face_score, age = get_meta(mat_path, db)

    gender_classes = []
    ages_classes = []
    imgs_data = []

    # Load the npz file
    for i in tqdm(range(len(face_score))):

        # Each Image is provided a score idicating the quality of the face in the image. 
        # Here we filter all scores that do not meet the threshold.
        if face_score[i] < min_score:
            continue

        # Some Images contain multiple faces here we're removing those images
        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue
        
        # All images of ages between 0 and 100
        if ~(0 <= age[i] <= 100):
            continue

        # Some images do not provide a gender
        if np.isnan(gender[i]):
            continue
        
        # Create the array of images to be stored in the npz file.
        gender_classes.append(int(gender[i]))
        ages_classes.append(age[i])
        
        # Grab the pixel data of each image and add it to the 
        img = cv2.imread(root_path + str(full_path[i][0]))
        imgs_data.append(cv2.resize(img, (img_size, img_size)))

    # Save the npz file
    np.savez(output_path, images=np.array(imgs_data), gender=np.array(gender_classes), age=np.array(ages_classes), img_size=img_size)

if __name__ == '__main__':
    main()
