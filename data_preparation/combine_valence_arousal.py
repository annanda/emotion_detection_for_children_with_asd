"""
author: Annanda Sousa
script for combining valence and arousal gold standard annotation into one csv file.
original files come form labels/arousal and labels/valence
the resulted csv file goes to labels/combined_valence_arousal
"""

import glob
import pathlib
import os.path
from scipy.io import arff
import pandas as pd


def merge_annotation_dataset(file_arousal, files_names_valence):
    name = file_arousal.split("/")[-1]
    file_valence = [file for file in files_names_valence if name in file][0]

    data_arousal = arff.loadarff(file_arousal)
    data_arousal = pd.DataFrame(data_arousal[0])

    data_valence = arff.loadarff(file_valence)
    data_valence = pd.DataFrame(data_valence[0])

    merged_annotation = data_valence.merge(data_arousal, how='left', on='frametime')
    path = pathlib.Path(__file__).parent.parent.absolute()
    path_arousal = os.path.join(path, 'labels', 'combined_valence_arousal')
    file_name = name.split(".")[0]
    merged_annotation.to_csv(f"{path_arousal}/{file_name}.csv", index=False)


def merge_valence_arousal():
    path = pathlib.Path(__file__).parent.parent.absolute()
    path_arousal = os.path.join(path, 'labels', 'arousal')
    path_valence = os.path.join(path, 'labels', 'valence')

    files_names_arousal = glob.glob(f"{path_arousal}/*.arff")
    files_names_valence = glob.glob(f"{path_valence}/*.arff")

    for file_arousal in files_names_arousal:
        merge_annotation_dataset(file_arousal, files_names_valence)


if __name__ == '__main__':
    merge_valence_arousal()
