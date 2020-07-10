""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
import functools
import os.path
from classifiers.emotion_detection_classifier import run_model_more_than_one_feature_type, run_model_one_feature_type
from data_preparation.combine_feature_annotation import call_merge_video_files
from data_preparation.concatenating_datasets import concatenate_dataset_files
from setup.conf import DATASET_FOLDER
from classifiers.late_fusion_layer import late_fusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd


def run_system(modality, features_type, model):
    if modality not in ['video', 'audio']:
        raise TypeError('Modality must be video or audio')
    if len(features_type) == 1:
        # accuracy, confusion_mtrx = run_model_one_feature_type(modality, features_type[0], model)
        predictions, y_test = run_model_one_feature_type(modality, features_type[0], model)
    else:
        # accuracy, confusion_mtrx = run_model_more_than_one_feature_type(modality, features_type, model)
        predictions, y_test = run_model_more_than_one_feature_type(modality, features_type, model)

    #  to not break the code yet.
    #####################################

    #####################################

    return predictions, y_test


def computing_modalities(modalities, features_type, model):
    predictions = []
    y_tests = []
    for modality in modalities:
        prediction, y_test = run_system(modality, features_type, model)
        predictions.append(prediction)
        y_tests.append(y_test)
    # predictions_2, y_test_2 = run_system(modality, features_type, model)

    predictions_multimodality = late_fusion(predictions[0], predictions[1])
    print(predictions_multimodality)
    # accuracy_1 = accuracy_score(y_test_1, predictions)
    # accuracy_2 = accuracy_score(y_test_2, predictions)
    #
    # confusion_mtrx_1 = confusion_matrix(y_test_1, predictions, labels=["blue", "green", "yellow", "red"])
    # confusion_mtrx_2 = confusion_matrix(y_test_2, predictions, labels=["blue", "green", "yellow", "red"])
    #
    # print(f'Accuracy of {model} model for {modality} modality using {features_type} as features: {accuracy_1}')
    # print('Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    # print(confusion_mtrx_1)


def prepare_data(modality, features_type):
    if modality not in ['video', 'audio']:
        raise TypeError('Modality must be video or audio')
    call_merge_video_files(modality, features_type)
    concatenate_dataset_files(modality, 'dev', features_type)
    concatenate_dataset_files(modality, 'train', features_type)


# def system_entry(modality, features_type, model, fusion_type):
def concatenate_annotation_test_predictions(predictions_multimodal, y_test_list):
    y_test = None
    return y_test


def call_multimodal_ed_system(data_entry):
    dfs = []
    #  TODO: get information of each modality to compute separetely
    modalities = list(input_data['modalities'].keys())
    for modality in modalities:
        features_type, model = prepate_entry_data(modality)
        prediction_1, y_test_1 = call_unimodal_ed_system(modality, features_type, model)

        # _, features_type, model = prepate_entry_data(data_entry)
        # prediction_2, y_test_2 = call_unimodal_ed_system(modality, features_type, model)

        dfs.append(prediction_1)
        dfs.append(y_test_1)

        # dfs.append(prediction_2)
        # dfs.append(y_test_2)

    # y_test_list.append(y_test_1)
    # y_test_list.append(y_test_2)

    merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='inner'), dfs)
    y_test = merged.iloc[:, 2:3]
    prediction_1 = merged.iloc[:, :1]
    prediction_2 = merged.iloc[:, 1:2]
    predictions_multimodal = late_fusion(prediction_1, prediction_2)

    return predictions_multimodal, y_test


def system_entry(data_entry):
    if not set(list(data_entry['modalities'].keys())).issubset(['video', 'audio']):
        raise TypeError('Modality must be video and/or audio')

    fusion_type = data_entry['fusion_type']
    if len(data_entry['modalities']) < 2:
        fusion_type = False

    if not fusion_type:
        modality = list(data_entry['modalities'].keys())[0]
        features_type, model = prepate_entry_data(modality)
        predictions, y_test = call_unimodal_ed_system(modality, features_type, model)
    else:
        predictions, y_test = call_multimodal_ed_system(data_entry)

    accuracy = accuracy_score(y_test, predictions)

    confusion_mtrx = confusion_matrix(y_test, predictions, labels=["blue", "green", "yellow", "red"])

    print(f'Accuracy of {model} model for {modality} modality using {features_type} as features: {accuracy}')
    print('Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    print(confusion_mtrx)


def call_unimodal_ed_system(modality, features_type, model):
    if len(features_type) == 1:
        path_to_check = f'{DATASET_FOLDER}/{modality}/{features_type[0]}_dev.csv'
        if not os.path.isfile(path_to_check):
            prepare_data(modality, features_type[0])
    predictions, y_test = run_system(modality, features_type, model)
    return predictions, y_test


#   computing modalities come here, if the number of modalities is bigger than one.


def prepate_entry_data(modality):
    # for modality in input_data['modalities'].keys():
    features_type = [key for key in input_data['modalities'][modality]['features_type'].keys() if
                     input_data['modalities'][modality]['features_type'][key] is True]
    model = input_data['modalities'][modality]['model']
    modality = modality

    return features_type, model


if __name__ == '__main__':
    # input_data = {
    #     'modalities': {
    #         'video': {
    #             'features_type': {'AU': True, 'appearance': True, 'BoVW': False, 'geometric': False},
    #             'model': 'SVM'
    #         },
    #         'audio': {
    #             'features_type': {'BoAW': True},
    #             'model': 'SVM'
    #         }
    #
    #     },
    #     'fusion_type': 'late_fusion'}
    input_data = {
        'modalities': {
            'video': {
                'features_type': {'AU': True, 'appearance': False, 'BoVW': False, 'geometric': False},
                'model': 'SVM'
            },
            'audio': {
                'features_type': {'BoAW': True},
                'model': 'SVM'
            }
        },
        'fusion_type': 'late_fusion'}
    system_entry(input_data)
    # prepate_entry_data(input_data)
#  so the idea is to transform the user's request into this dictionary that the system will use,
#  So, from this point, I can deal with the data in this format - inside the ED system.
