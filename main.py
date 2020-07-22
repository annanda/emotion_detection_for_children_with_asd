""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
import functools
import os.path
from classifiers.emotion_detection_classifier import run_model_more_than_one_feature_type, run_model_one_feature_type
from data_preparation.preparing_datasets import prepare_data
from setup.conf import DATASET_FOLDER, EMOTION_ANNOTATION_FILE
from classifiers.late_fusion_layer import late_fusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd


def run_system(modality, features_type, model):
    if modality not in ['video', 'audio', 'physio']:
        raise TypeError('Modality must be video, physio or audio')
    if len(features_type) == 1:
        prediction_and_true_value = run_model_one_feature_type(modality, features_type[0], model)
    else:
        prediction_and_true_value = run_model_more_than_one_feature_type(modality, features_type, model)

    return prediction_and_true_value


def call_multimodal_ed_system(data_entry):
    dfs = []
    modalities = list(data_entry['modalities'].keys())
    for modality in modalities:
        features_type, model = prepare_entry_data(modality)
        prediction_and_true_value = call_unimodal_ed_system(modality, features_type, model)
        dfs.append(prediction_and_true_value)

    merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='outer'), dfs)
    emotion_annotation = pd.read_csv(EMOTION_ANNOTATION_FILE)
    merged = pd.merge(merged, emotion_annotation, on='frametime', how='inner')
    y_test = merged.iloc[:, -1:]
    list_of_predictions = []
    list_of_predictions.append(merged['predictions_x'].tolist())
    list_of_predictions.append(merged['predictions_y'].tolist())
    try:
        list_of_predictions.append(merged['predictions'].tolist())
    except KeyError:
        pass

    predictions_multimodal = late_fusion(list_of_predictions)

    return predictions_multimodal, y_test


def system_entry(data_entry):
    if not set(list(data_entry['modalities'].keys())).issubset(['video', 'audio', 'physio']):
        raise TypeError('Modality must be video, physio and/or audio')

    fusion_type = data_entry['fusion_type']
    if len(data_entry['modalities']) < 2:
        fusion_type = False

    if not fusion_type:
        modality = list(data_entry['modalities'].keys())[0]
        features_type, model = prepare_entry_data(modality)
        predictions_and_y_test = call_unimodal_ed_system(modality, features_type, model)
        predictions = predictions_and_y_test['predictions'].tolist()
        y_test = predictions_and_y_test['emotion_zone'].tolist()
    else:
        predictions, y_test = call_multimodal_ed_system(data_entry)

    accuracy = accuracy_score(y_test, predictions)
    confusion_mtrx = confusion_matrix(y_test, predictions, labels=["blue", "green", "yellow", "red"])

    print(f'Processing the input: {data_entry}')
    print('###########################################')
    print(f'Accuracy: {accuracy}')
    print('Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    print(confusion_mtrx)


def call_unimodal_ed_system(modality, features_type, model):
    if len(features_type) == 1:
        path_to_check = f'{DATASET_FOLDER}/{modality}/{features_type[0]}_dev.csv'
        if not os.path.isfile(path_to_check):
            prepare_data(modality, features_type[0])
    prediction_and_true_value = run_system(modality, features_type, model)
    return prediction_and_true_value


def prepare_entry_data(modality):
    features_type = [key for key in input_data['modalities'][modality]['features_type'].keys() if
                     input_data['modalities'][modality]['features_type'][key] is True]
    model = input_data['modalities'][modality]['model']

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
                'features_type': {'BoAW': True, 'DeepSpectrum': False, 'eGeMAPSfunct': False},
                'model': 'SVM'
            },
            'physio': {
                'features_type': {'HRHRV': True},
                'model': 'SVM'
            }
        },
        'fusion_type': 'late_fusion'}
    system_entry(input_data)
#  so the idea is to transform the user's request into this dictionary that the system will use,
#  So, from this point, I can deal with the data in this format - inside the ED system.
