"""
author: Annanda Sousa
Script to map values of arousal and valence into the 4 emotion zones.
"""
import numpy as np
import glob
import pandas as pd
import pathlib
import os.path

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


def map_emotion_zones(path_to_combined_files):
    files = glob.glob(f"{path_to_combined_files}/*.csv")
    for file in files:
        new_rows = []
        dataframe_annotation = pd.read_csv(file)
        for row in dataframe_annotation.itertuples():
            valence = row.GoldStandardValence
            arousal = row.GoldStandardArousal
            emotion = get_emotion_zone(valence, arousal)
            row = [row.frametime, emotion]
            new_rows.append(row)
        new_dataframe = pd.DataFrame(new_rows, columns=['frametime', 'emotion_zone'])
        path = pathlib.Path(__file__).parent.parent.absolute()
        path_annotation_emotions = os.path.join(path, 'labels', 'emotion_zones', 'emotion_names')
        file_name = file.split("/")[-1]
        new_dataframe.to_csv(f"{path_annotation_emotions}/{file_name}", index=False)


if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.parent.absolute()
    path_annotation_emotions = os.path.join(path, 'labels', 'combined_valence_arousal')
    map_emotion_zones(path_annotation_emotions)
    # map_emotion_zones("/Users/user/PycharmProjects/emotion_detection_system/labels/combined_valence_arousal")
