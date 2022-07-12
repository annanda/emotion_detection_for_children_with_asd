import os.path
import functools
import glob
import pandas as pd
from scipy.io import arff
from setup.conf import MAIN_FOLDER, DATASET_FOLDER

path_annotation_emotions = os.path.join(MAIN_FOLDER, 'labels', 'emotion_zones', 'emotion_names')
list_file_annotation_emotions = glob.glob(f"{path_annotation_emotions}/*.csv")


def concatenate_dataset_files(modality, dataset_type, features_type):
    """
    concatenate all train or dev files into one big dataset file
    dataset_type must be 'train' or 'dev'
    """
    if dataset_type not in ['train', 'dev']:
        raise TypeError('dataset_type must be train or dev')
    files = glob.glob(f'{DATASET_FOLDER}/{modality}/{features_type}/{dataset_type}_*.csv')

    if not files:
        raise ValueError('You do not have train/dev files. You need to define Train and Dev division for your dataset.')

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    concated = pd.concat(dfs)
    concated.to_csv(f'{DATASET_FOLDER}/{modality}/{features_type}_{dataset_type}.csv', index=False)


def merge_different_features_type_together(modality, dataset_type, features_type_list):
    """
    dataset_type must be 'train' or 'dev'
    In here, for each train/dev file, I create one train/dev file with different features type merged together,
     i.e. a dataset with more features.
    """

    for features_type in features_type_list:
        path_to_check = f'{DATASET_FOLDER}/{modality}/{features_type}_dev.csv'
        if not os.path.isfile(path_to_check):
            produce_one_feature_type(modality, features_type)

    files = glob.glob(f'{DATASET_FOLDER}/{modality}/{features_type_list[0]}/{dataset_type}_*.csv')

    for file in files[:]:
        dfs = []
        file_name = file.split('/')[-1]
        for i, feature_type in enumerate(features_type_list):
            other_df_path = f'{DATASET_FOLDER}/{modality}/{features_type_list[i]}/{file_name}'
            other_df = pd.read_csv(other_df_path)
            dfs.append(other_df)
        merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='inner'), dfs)
        merged.to_csv(f'{DATASET_FOLDER}/{modality}/temp/{file_name}', index=False)


def produce_more_than_one_features_type(modality, feature_type_lst):
    merge_different_features_type_together(modality, 'train', feature_type_lst)
    merge_different_features_type_together(modality, 'dev', feature_type_lst)
    concatenate_dataset_files(modality, 'train', 'temp')
    concatenate_dataset_files(modality, 'dev', 'temp')


def produce_one_feature_type(modality, features_type):
    generate_feature_files(modality, features_type)
    concatenate_dataset_files(modality, 'dev', features_type)
    concatenate_dataset_files(modality, 'train', features_type)


def generate_one_feature_file(modality, features_file_name, feature_type):
    file_name_suffix = features_file_name.split('/')[-1]
    if '_3.0_0' in file_name_suffix:
        file_name_suffix = file_name_suffix.split('_3.0_0')[0]
    elif '_0.0_0' in file_name_suffix:
        file_name_suffix = file_name_suffix.split('_0.0_0')[0]
    elif '_2.0_0' in file_name_suffix:
        file_name_suffix = file_name_suffix.split('_2.0_0')[0]
    data_features = arff.loadarff(features_file_name)
    data_features_df = pd.DataFrame(data_features[0])
    data_features_df.dropna(inplace=True)
    data_features_df['frametime'] = file_name_suffix + '___' + data_features_df['frametime'].astype(str)
    path_dataset = os.path.join(MAIN_FOLDER, 'dataset', f'{modality}', f'{feature_type}')
    data_features_df.to_csv(f"{path_dataset}/{file_name_suffix}.csv", index=False)


def generate_feature_files(modality, feature_type):
    """
    """
    path_features_train = os.path.join(MAIN_FOLDER, 'features', f'{modality}', f'{feature_type}')
    list_files_features = glob.glob(f"{path_features_train}/*.arff")
    list_files_features.sort()
    for file in list_files_features:
        generate_one_feature_file(modality, file, feature_type)


if __name__ == '__main__':
    features_type = 'BoVW'
    feature_type_list = ['AU', 'appearance', 'BoVW']
    produce_more_than_one_features_type(feature_type_list)
