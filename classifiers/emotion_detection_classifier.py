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
    # for person specific model
    if individual_model:
        individual_model = 'individuals'
    # for person-independent model
    else:
        individual_model = 'cross_individuals'
    feature_folder = os.path.join(DATASET_FOLDER, dataset_split_type, individual_model, modality,
                                  feature_type)
    # Data from a specific session
    if len(session_number) == 1:
        folder_dataset = os.path.join(feature_folder, session_number[0])
    # all the data from one participant, i.e., two sessions
    else:
        folder_dataset_01 = os.path.join(feature_folder, session_number[0])
        folder_dataset_02 = os.path.join(feature_folder, session_number[1])
        folder_dataset = [folder_dataset_01, folder_dataset_02]
    train_dataset = get_dataset_split(folder_dataset, 'train')
    dev_dataset = get_dataset_split(folder_dataset, 'dev')
    test_dataset = get_dataset_split(folder_dataset, 'test')

    # getting each of the splits for the dataset
    x = train_dataset.iloc[:, 4:]
    y = train_dataset['emotion_zone']

    # the model is not currently using dev set!
    x_dev = dev_dataset.iloc[:, 4:]
    y_dev = dev_dataset['emotion_zone']

    x_test = test_dataset.iloc[:, 4:]
    y_test = test_dataset['emotion_zone']

    # Selecting the model and training it
    if model == 'SVM':
        clf = svm.SVC(probability=True)

    print(f'Training the model: {model}')
    print('###########################################')
    print('.\n.\n.\n.')

    clf.fit(x, y)
    # clf.classes_ return the labels - blue, green, red, yellow
    prediction_probability = clf.predict_proba(x_test)
    indexes = list(y_test.index)
    # organising the prediction results with the labels
    prediction_probability_df = pd.DataFrame(prediction_probability, columns=['blue', 'green', 'red', 'yellow'],
                                             index=indexes)
    return prediction_probability_df, y_test


def get_dataset_split(dataset_folder, split_type):
    split_dfs = []
    if isinstance(dataset_folder, list):
        type_files_in_folder_01 = [type_file for type_file in os.listdir(dataset_folder[0]) if split_type in type_file]
        type_files_in_folder_02 = [type_file for type_file in os.listdir(dataset_folder[1]) if
                                   split_type in type_file]
        for type_file in type_files_in_folder_01:
            train_dataset_df = pd.read_pickle(os.path.join(dataset_folder[0], type_file))
            split_dfs.append(train_dataset_df)
        for type_file in type_files_in_folder_02:
            train_dataset_df = pd.read_pickle(os.path.join(dataset_folder[1], type_file))
            split_dfs.append(train_dataset_df)
    else:
        type_files_in_folder = [type_file for type_file in os.listdir(dataset_folder) if split_type in type_file]
        for type_file in type_files_in_folder:
            train_dataset_df = pd.read_pickle(os.path.join(dataset_folder, type_file))
            split_dfs.append(train_dataset_df)
    concated_split_dataset = pd.concat(split_dfs)
    return concated_split_dataset


# predictions_array = clf.predict(x_test)
# prediction_and_true_value = y_test.assign(predictions=predictions_array)
# return prediction_and_true_value


# def run_model_more_than_one_feature_type(modality, feature_type_list, model):
def run_model_more_than_one_feature_type(session_number, dataset_split_type, individual_model,
                                         modality, features_type_list, model):
    produce_more_than_one_features_type(session_number, dataset_split_type, individual_model, modality,
                                        features_type_list)
    # produce_more_than_one_features_type(modality, features_type_list)

    prediction_and_true_value = run_model_one_feature_type(session_number, dataset_split_type, individual_model,
                                                           modality, features_type_list, model)
    # prediction_and_true_value = run_model_one_feature_type(modality, 'temp', model)

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
        predictions_probability, y_test = run_model_one_feature_type(session_number, dataset_split_type,
                                                                     individual_model,
                                                                     modality, features_type[0], model)
    else:
        predictions_probability, y_test = run_model_more_than_one_feature_type(session_number, dataset_split_type,
                                                                               individual_model,
                                                                               modality, features_type, model)
        prediction_and_true_value = run_model_more_than_one_feature_type(modality, features_type, model)

    return predictions_probability, y_test


def get_predicted_class(probability_vector):
    emotion_class = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow'}
    result = np.where(probability_vector == np.amax(probability_vector))
    emotion_index = result[0][0]
    return emotion_class[emotion_index]


def get_final_label_prediction_array(predictions_probabilities):
    predictions = []
    for _, row in predictions_probabilities.iterrows():
        label = get_predicted_class(np.array(row[['blue', 'green', 'red', 'yellow']]))
        predictions.append(label)
    return predictions
