import os.path
import glob
import pandas as pd
from setup.conf import MAIN_FOLDER, DATASET_VIDEO_AU_FOLDER


def concatenate_video_au_files(dataset_type):
    """
    dataset_type can be 'train' or 'dev'
    """
    files = glob.glob(f'{DATASET_VIDEO_AU_FOLDER}/{dataset_type}_*.csv')

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    concated = pd.concat(dfs)
    concated.to_csv(f'{MAIN_FOLDER}/dataset/video/AU_{dataset_type}.csv')


if __name__ == '__main__':
    concatenate_video_au_files('dev')
