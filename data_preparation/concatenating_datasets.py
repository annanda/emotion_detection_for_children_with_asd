import os.path
import functools
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


def concatenate_different_features_type_dataset(dataset_type, features_type_list):
    """
    dataset_type must be 'train' or 'dev'
    """
    files = glob.glob(f'{DATASET_VIDEO_FOLDER}/{features_type_list[0]}/{dataset_type}_*.csv')

    for file in files:
        dfs = []
        file_name = file.split('/')[-1]
        df = pd.read_csv(file).iloc[:, :-1]
        other_df_path = f'{DATASET_VIDEO_FOLDER}/{features_type_list[1]}/{file_name}'
        other_df = pd.read_csv(other_df_path)
        dfs.append(df)
        dfs.append(other_df)
        merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='inner'), dfs)
        # print(merged)
        merged.to_csv(f'{DATASET_VIDEO_FOLDER}/temp/{file_name}', index=False)


# def concatenate_annotated_datasets(feature_type_list):
#     # esse chama a funcao acima
#     dfs = []
#     for i, feature_type in enumerate(feature_type_list):
#         if i < len(feature_type_list) - 1:
#             df = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/{feature_type}_train.csv').iloc[:, 1:-1]
#             dfs.append(df)
#         else:
#             last_df = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/{feature_type}_train.csv').iloc[:, 1:]
#             dfs.append(last_df)
#     # merged = pd.merge(data_features_df, data_emotion_annotation_df, how='inner', on='frametime')
#     # merged = pd.concat(dfs, axis=1)
#     merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='inner'), dfs)
#     working_dataset = merged
#     # return working_dataset


def producing_more_than_one_features_type(feature_type_lst):
    concatenate_different_features_type_dataset('train', feature_type_lst)
    concatenate_different_features_type_dataset('dev', feature_type_lst)
    concatenate_video_au_files('train', 'temp')
    concatenate_video_au_files('dev', 'temp')


if __name__ == '__main__':
    # features_type = 'appearance'
    # concatenate_video_au_files('dev', features_type)
    feature_type_list = ['AU', 'appearance']
    producing_more_than_one_features_type(feature_type_list)
