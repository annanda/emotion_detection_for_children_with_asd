import os.path
import functools
import glob
import pandas as pd
from setup.conf import MAIN_FOLDER, DATASET_FOLDER
from data_preparation.combine_feature_annotation import call_merge_modality_files, merge_for_all_files


def concatenate_dataset_files(modality, dataset_type, features_type):
    """
    concatenate all train or dev files into one big dataset file
    dataset_type must be 'train' or 'dev'
    """
    if dataset_type not in ['train', 'dev']:
        raise TypeError('dataset_type must be train or dev')
    files = glob.glob(f'{DATASET_FOLDER}/{modality}/{features_type}/{dataset_type}_*.csv')

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    concated = pd.concat(dfs)
    concated.to_csv(f'{DATASET_FOLDER}/{modality}/{features_type}_{dataset_type}.csv', index=False)


def concatenate_different_features_type_dataset(modality, dataset_type, features_type_list):
    """
    dataset_type must be 'train' or 'dev'
    """

    for features_type in features_type_list:
        path_to_check = f'{DATASET_FOLDER}/{modality}/{features_type}_dev.csv'
        if not os.path.isfile(path_to_check):
            prepare_data(modality, features_type)

    files = glob.glob(f'{DATASET_FOLDER}/{modality}/{features_type_list[0]}/{dataset_type}_*.csv')

    for file in files[:]:
        dfs = []
        file_name = file.split('/')[-1]
        # df = pd.read_csv(file).iloc[:, :-1]
        # dfs.append(df)
        for i, feature_type in enumerate(features_type_list):
            if i < len(features_type_list) - 1:
                other_df_path = f'{DATASET_FOLDER}/{modality}/{features_type_list[i]}/{file_name}'
                other_df = pd.read_csv(other_df_path).iloc[:, :-1]
                dfs.append(other_df)
            else:
                last_df_path = f'{DATASET_FOLDER}/{modality}/{features_type_list[-1]}/{file_name}'
                last_df_path = pd.read_csv(last_df_path)
                dfs.append(last_df_path)

        merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='inner'), dfs)
        # print(merged)
        merged.to_csv(f'{DATASET_FOLDER}/{modality}/temp/{file_name}', index=False)


def producing_more_than_one_features_type(modality, feature_type_lst):
    concatenate_different_features_type_dataset(modality, 'train', feature_type_lst)
    concatenate_different_features_type_dataset(modality, 'dev', feature_type_lst)
    concatenate_dataset_files(modality, 'train', 'temp')
    concatenate_dataset_files(modality, 'dev', 'temp')


def prepare_data(modality, features_type):
    if modality not in ['video', 'audio', 'physio']:
        raise TypeError('Modality must be video, audio or physio')
    call_merge_modality_files(modality, features_type)
    concatenate_dataset_files(modality, 'dev', features_type)
    concatenate_dataset_files(modality, 'train', features_type)


if __name__ == '__main__':
    features_type = 'BoVW'
    # concatenate_video_files('dev', features_type)
    feature_type_list = ['AU', 'appearance', 'BoVW']
    producing_more_than_one_features_type(feature_type_list)
