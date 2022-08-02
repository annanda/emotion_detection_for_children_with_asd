import functools
import os

import pandas as pd
import numpy as np
from setup.conf import DATASET_FOLDER, EMOTION_ANNOTATION_FILE
from sklearn import svm

from sklearn.model_selection import train_test_split

from data_preparation.dataset_preparation import produce_more_than_one_features_type
from data_preparation.dataset_balancing import balance_dataset_undersampling
from classifiers.late_fusion_layer import late_fusion
from data_preparation.dataset_preparation import produce_one_feature_type


def run_model_one_feature_type(session_number, dataset_split_type, individual_model, modality, feature_type, model):
    if individual_model:
        individual_model = 'individuals'
    else:
        individual_model = 'cross_individuals'
    feature_folder = os.path.join(DATASET_FOLDER, dataset_split_type, individual_model, modality,
                                  feature_type)
    # Data from a specific session
    if len(session_number) == 1:
        folder_dataset = os.path.join(feature_folder, session_number[0])
    else:
        folder_dataset_01 = os.path.join(feature_folder, session_number[0])
        folder_dataset_02 = os.path.join(feature_folder, session_number[1])
        folder_dataset = [folder_dataset_01, folder_dataset_02]
    train_dataset = get_dataset(folder_dataset, 'train')
    dev_dataset = get_dataset(folder_dataset, 'dev')
    test_dataset = get_dataset(folder_dataset, 'test')

###################################################################
    # I STOPPED HERE

    x = train_dataset.iloc[:, 1:-1]
    y = train_dataset['emotion_zone']

    x_dev_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_dev.csv')
    x_dev_dataset_annotated = pd.merge(x_dev_dataset, emotion_annotation, how='inner', on='frametime')
    y_dev_dataset = x_dev_dataset_annotated[['frametime', 'emotion_zone']]
    x_dev_dataset = x_dev_dataset_annotated.iloc[:, 1:-1]
    # the model is not currently using dev set!
    x_dev, x_test, y_dev, y_test = train_test_split(x_dev_dataset, y_dev_dataset, test_size=0.2)

    if model == 'SVM':
        clf = svm.SVC(probability=True)
    clf.fit(x, y)
    # clf.classes_ return the labels - blue, green, red, yellow
    prediction_probability = clf.predict_proba(x_test)
    indexes = list(y_test.index)
    prediction_probability_df = pd.DataFrame(prediction_probability, columns=['blue', 'green', 'red', 'yellow'],
                                             index=indexes)
    prediction_and_true_value = y_test.assign(blue=prediction_probability_df['blue'],
                                              green=prediction_probability_df['green'],
                                              red=prediction_probability_df['red'],
                                              yellow=prediction_probability_df['yellow'])
    return prediction_and_true_value


def get_dataset(dataset_folder, split_type):
    # Data from a specific session
    # if len(session_number) == 1:
    #     folder_dataset = os.path.join(feature_folder, session_number[0])
    #     train_files_in_folder = [train_file for train_file in os.listdir(folder_dataset) if 'train' in train_file]
    #     train_dfs = []
    #     for train_file in train_files_in_folder:
    #         train_dataset_df = pd.read_pickle(os.path.join(folder_dataset, train_file))
    #         train_dfs.append(train_dataset_df)
    #     # train_dataset = pd.concat(train_dfs)
    # # All data for a participant, i.e., two sessions
    # else:
    #     folder_dataset_01 = os.path.join(feature_folder, session_number[0])
    #     folder_dataset_02 = os.path.join(feature_folder, session_number[1])
    #     train_files_in_folder_01 = [train_file for train_file in os.listdir(folder_dataset_01) if 'train' in train_file]
    #     train_files_in_folder_02 = [train_file for train_file in os.listdir(folder_dataset_02) if
    #                                 'train' in train_file]
    #     train_dfs = []
    #     for train_file in train_files_in_folder_01:
    #         train_dataset_df = pd.read_pickle(os.path.join(folder_dataset_01, train_file))
    #         train_dfs.append(train_dataset_df)
    #     for train_file in train_files_in_folder_02:
    #         train_dataset_df = pd.read_pickle(os.path.join(folder_dataset_02, train_file))
    #         train_dfs.append(train_dataset_df)
    # train_dataset = pd.concat(train_dfs)
    dataframe = [None]
    if dataset_folder is str:
        pass

    return dataframe


# predictions_array = clf.predict(x_test)
# prediction_and_true_value = y_test.assign(predictions=predictions_array)
# return prediction_and_true_value


def run_model_more_than_one_feature_type(modality, feature_type_list, model):
    produce_more_than_one_features_type(modality, feature_type_list)
    prediction_and_true_value = run_model_one_feature_type(modality, 'temp', model)
    return prediction_and_true_value


def call_multimodal_ed_system(data_entry):
    dfs = []
    modalities = list(data_entry['modalities'].keys())
    for modality in modalities:
        features_type, model = get_features_and_model(modality, data_entry)
        prediction_and_true_value = call_unimodal_ed_system(modality, features_type, model)
        dfs.append(prediction_and_true_value)

    merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='outer'), dfs)
    emotion_annotation = pd.read_csv(EMOTION_ANNOTATION_FILE)
    merged = pd.merge(merged, emotion_annotation, on='frametime', how='inner')
    y_test = merged.iloc[:, -1:]
    y_test.columns = ['emotion_zone']
    predictions_multimodal = late_fusion(merged)
    predictions_multimodal_and_true_value = y_test.assign(
        blue=predictions_multimodal['blue'],
        green=predictions_multimodal['green'],
        red=predictions_multimodal['red'],
        yellow=predictions_multimodal['yellow']
    )
    return predictions_multimodal_and_true_value


def get_features_model_and_modality(input_data):
    modality = list(input_data['modalities'].keys())[0]

    # Getting all features types that are set to True on the configuration_data
    features_type = [key for key in input_data['modalities'][modality]['features_type'].keys() if
                     input_data['modalities'][modality]['features_type'][key] is True]

    model = input_data['modalities'][modality]['model']

    return features_type, model, modality


def call_unimodal_ed_system(session_number, dataset_split_type, individual_model, modality, features_type, model):
    """
    :param session_number: a list of strings, with the sessions to consider the data
    :param modality: a string with the modality type for the model
    :param features_type: a list of strings, with the feature types to consider
    :param model: a string with the type of the model to consider
    :return:
    """
    if len(features_type) == 1:
        prediction_and_true_value = run_model_one_feature_type(session_number, dataset_split_type, individual_model,
                                                               modality, features_type[0], model)
    else:
        prediction_and_true_value = run_model_more_than_one_feature_type(modality, features_type, model)

    return prediction_and_true_value


def get_predicted_class(probability_vector):
    emotion_class = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow'}
    result = np.where(probability_vector == np.amax(probability_vector))
    emotion_index = result[0][0]
    return emotion_class[emotion_index]


def get_final_label_prediction_array(predictions_and_y_test):
    predictions = []
    for index, row in predictions_and_y_test.iterrows():
        label = get_predicted_class(np.array(row[['blue', 'green', 'red', 'yellow']]))
        predictions.append(label)
    return predictions
