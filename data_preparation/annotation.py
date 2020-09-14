"""
author: Annanda Sousa
Script to map values of arousal and valence into the 4 emotion zones.
"""
import numpy as np
import os.path
import glob
import pathlib
import os.path
from scipy.io import arff
import pandas as pd
from setup.conf import MAIN_FOLDER

path_annotation_emotions = os.path.join(MAIN_FOLDER, 'labels', 'emotion_zones', 'emotion_names')
list_file_annotation_emotions = glob.glob(f"{path_annotation_emotions}/*.csv")
emotion_zone = {
    'blue': np.array([1, 0, 0, 0]),
    'green': np.array([0, 1, 0, 0]),
    'yellow': np.array([0, 0, 1, 0]),
    'red': np.array([0, 0, 0, 1])
}


def get_emotion_zone(valence, arousal):
    """
    each emotion zone is represented by an np array of 4 dimension
    on each orthogonal the emotion zone is the one located on the right of the orthogonal in question.
    the point 0,0 corresponds to green zone.
    """
    emotion = None
    if arousal == 0 and valence == 0:
        emotion = 'green'
    elif arousal > 0:
        if valence >= 0:
            emotion = 'yellow'
        else:
            emotion = 'red'
    elif arousal < 0:
        if valence > 0:
            emotion = 'green'
        else:
            emotion = 'blue'
    elif arousal == 0:
        if valence > 0:
            emotion = 'green'
        else:
            emotion = 'red'
    elif valence == 0:
        if arousal > 0:
            emotion = 'yellow'
        else:
            emotion = 'blue'
    return emotion


def get_emotion_zone_advanced(valence, arousal):
    threshold = 0.25
    if threshold > valence >= threshold:
        return None
    if threshold > arousal >= threshold:
        return None
    return get_emotion_zone(valence, arousal)


def map_emotion_zones(path_to_combined_files):
    files = glob.glob(f"{path_to_combined_files}/*.csv")
    for file in files:
        new_rows = []
        dataframe_annotation = pd.read_csv(file)
        for row in dataframe_annotation.itertuples():
            valence = row.GoldStandardValence
            arousal = row.GoldStandardArousal
            # emotion = get_emotion_zone(valence, arousal)
            emotion = get_emotion_zone_advanced(valence, arousal)
            row = [row.frametime, emotion]
            new_rows.append(row)
        new_dataframe = pd.DataFrame(new_rows, columns=['frametime', 'emotion_zone'])
        path = pathlib.Path(__file__).parent.parent.absolute()
        path_annotation_emotions = os.path.join(path, 'labels', 'emotion_zones', 'emotion_names')
        file_name = file.split("/")[-1]
        file_name = 'advanced_annotation_' + file_name
        new_dataframe.to_csv(f"{path_annotation_emotions}/{file_name}", index=False)


def create_single_annotation_file():
    dfs_emotions = []
    for file in list_file_annotation_emotions:
        file_name_suffix = file.split('/')[-1]
        file_name_suffix = file_name_suffix.split('.csv')[0]
        data_emotion_annotation_df = pd.read_csv(file)
        data_emotion_annotation_df['frametime'] = file_name_suffix + '___' + data_emotion_annotation_df[
            'frametime'].astype(str)
        dfs_emotions.append(data_emotion_annotation_df)
    result = pd.concat(dfs_emotions)
    result.to_csv(path_annotation_emotions + '/emotions_annotation.csv', index=False)


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
    path = pathlib.Path(__file__).parent.parent.absolute()
    path_annotation_emotions = os.path.join(path, 'labels', 'combined_valence_arousal')
    map_emotion_zones(path_annotation_emotions)
    # map_emotion_zones("/Users/user/PycharmProjects/emotion_detection_system/labels/combined_valence_arousal")
