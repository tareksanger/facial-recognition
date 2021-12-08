import os
from glob import glob


def parse_utk_meta(data_dir: str):

    filename_list =[]
    age_classes = []
    sex_classes = []
    race_classes = []    

    for filepath in glob(os.path.join(data_dir, "*.jpg")):

        try:
            _, filename = os.path.split(filepath)
            name, ext = os.path.splitext(filename)
            age, sex, race, _ = name.split("_")

            filename_list.append(filename)
            age_classes.append(int(age))
            sex_classes.append(int(sex))
            race_classes.append(int(race))
        
        except Exception as e:
            continue

    return filename_list, age_classes, sex_classes, race_classes


def get_age_class(age):
        if 0 <= age <= 3:
            return 0
        elif 4 <= age <= 6:
            return 1
        
        elif 7 <= age <= 12:
            return 2
        
        elif 13 <= age <= 20:
            return 3

        elif 21 <= age <= 24:
            return 4
        
        elif 25 <= age <= 32:
            return 5
        
        elif 33 <= age <= 37:
            return 6

        elif 38 <= age <= 43:
            return 7

        elif 44 <= age <= 48:
            return 8

        elif 49 <= age <= 59:
            return 9

        elif 60 <= age:
            return 10
        