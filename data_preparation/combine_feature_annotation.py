"""
author: Annanda Sousa
script to generate a CSV with the combined values of features and annotation for each 'frametime' of the dataset.
"""
import pathlib
import os.path
import pandas as pd
from scipy.io import arff
import glob
from setup.conf import MAIN_FOLDER

path_annotation_emotions = os.path.join(MAIN_FOLDER, 'labels', 'emotion_zones', 'emotion_names')
list_file_annotation_emotions = glob.glob(f"{path_annotation_emotions}/*.csv")


def generating_features_in_csv(modality, file_features, list_file_name_emotions, feature_type):
    file_name_suffix = file_features.split('/')[-1]
    if '_3.0_0' in file_name_suffix:
        file_name_suffix = file_name_suffix.split('_3.0_0')[0]
    elif '_0.0_0' in file_name_suffix:
        file_name_suffix = file_name_suffix.split('_0.0_0')[0]
    elif '_2.0_0' in file_name_suffix:
        file_name_suffix = file_name_suffix.split('_2.0_0')[0]
    data_features = arff.loadarff(file_features)
    data_features_df = pd.DataFrame(data_features[0])
    data_features_df.dropna(inplace=True)
    data_features_df['frametime'] = file_name_suffix + '___' + data_features_df['frametime'].astype(str)
    path_dataset = os.path.join(MAIN_FOLDER, 'dataset', f'{modality}', f'{feature_type}')
    data_features_df.to_csv(f"{path_dataset}/{file_name_suffix}.csv", index=False)


def merge_for_all_files(modality, list_file_features, list_file_annotation, feature_type):
    for list_train in list_file_features:
        generating_features_in_csv(modality, list_train, list_file_annotation, feature_type)


def call_merge_modality_files(modality, feature_type, list_file_annotation=list_file_annotation_emotions):
    """
    calling the function merge_for_all_files() for video modality - features Action Units (AU)
    """
    path_features_train = os.path.join(MAIN_FOLDER, 'features', f'{modality}', f'{feature_type}')
    list_file_features_train = glob.glob(f"{path_features_train}/*.arff")
    # eliminating the test files from the list, because we don't have annotation for test files.
    list_file_features_train = [file for file in list_file_features_train if 'test' not in file]
    list_file_features_train.sort()
    merge_for_all_files(modality, list_file_features_train, list_file_annotation, feature_type)


def create_annotation_file():
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


if __name__ == '__main__':
    # path_annotation_emotions = os.path.join(MAIN_FOLDER, 'labels', 'emotion_zones', 'emotion_names')
    # list_file_annotation_emotions = glob.glob(f"{path_annotation_emotions}/*.csv")
    # feature_type is part of the set {'AU', 'appearance'}
    # feature_type = 'AU'
    # feature_type = 'geometric'
    # modality = 'video'
    # call_merge_video_files(modality, feature_type, list_file_annotation_emotions)
    create_annotation_file()
