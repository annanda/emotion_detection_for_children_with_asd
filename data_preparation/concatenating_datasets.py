import os.path
import glob
import pandas as pd
from setup.conf import MAIN_FOLDER, DATASET_VIDEO_FOLDER


def concatenate_video_au_files(dataset_type, features_type):
    """
    dataset_type must be 'train' or 'dev'
    """
    files = glob.glob(f'{DATASET_VIDEO_FOLDER}/{features_type}/{dataset_type}_*.csv')

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    concated = pd.concat(dfs)
    concated.to_csv(f'{DATASET_VIDEO_FOLDER}/{features_type}_{dataset_type}.csv')


if __name__ == '__main__':
    features_type = 'appearance'
    concatenate_video_au_files('dev', features_type)
