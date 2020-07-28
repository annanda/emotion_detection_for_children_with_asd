""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
from classifiers.emotion_detection_classifier import get_features_and_model, call_multimodal_ed_system, \
    call_unimodal_ed_system, get_final_label_prediction_array

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np


def run_system(data_entry):
    if not set(list(data_entry['modalities'].keys())).issubset(['video', 'audio', 'physio']):
        raise TypeError('Modality must be video, physio and/or audio')

    fusion_type = data_entry['fusion_type']
    if len(data_entry['modalities']) < 2:
        fusion_type = False

    if not fusion_type:
        modality = list(data_entry['modalities'].keys())[0]
        features_type, model = get_features_and_model(modality, data_entry)
        predictions_and_y_test = call_unimodal_ed_system(modality, features_type, model)
        # predictions = predictions_and_y_test['predictions'].tolist()
        predictions = get_final_label_prediction_array(predictions_and_y_test)
    else:
        predictions_and_y_test = call_multimodal_ed_system(data_entry)
        predictions = get_final_label_prediction_array(predictions_and_y_test)

    y_test = predictions_and_y_test['emotion_zone'].tolist()
    accuracy, confusion_mtrx = calculate_evaluation_metrics(predictions, y_test)
    return accuracy, confusion_mtrx


def calculate_evaluation_metrics(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    confusion_mtrx = confusion_matrix(y_test, predictions, labels=["blue", "green", "yellow", "red"])
    return accuracy, confusion_mtrx


def print_results(accuracy, confusion_mtrx, data_entry, is_mean=False, times=None):
    print(f'Processing the input: {data_entry}')
    print('###########################################')
    if is_mean:
        print(f'Average of accuracy by running system {times} times: {accuracy}')
        print('Last computed Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    else:
        print(f'Accuracy: {accuracy}')
        print('Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    print(confusion_mtrx)


def run_x_times(times, data_entry):
    accuracy_values = []
    for _ in range(times):
        accuracy, confusion_mtrx = run_system(data_entry)
        accuracy_values.append(accuracy)

    accuracy_values = np.array(accuracy_values)
    print(accuracy_values)
    mean_accuracy = np.mean(accuracy_values)
    print_results(mean_accuracy, confusion_mtrx, data_entry, is_mean=True, times=times)


if __name__ == '__main__':
    input_data = {
        'modalities': {
            'video': {
                'features_type': {'AU': True, 'appearance': False, 'BoVW': False, 'geometric': False},
                'model': 'SVM'
            },
            # 'audio': {
            #     'features_type': {'BoAW': True, 'DeepSpectrum': False, 'eGeMAPSfunct': False},
            #     # eGeMAPSfunct feature_type can only be used alone
            #     'model': 'SVM'
            # },
            # 'physio': {
            #     'features_type': {'HRHRV': True},
            #     'model': 'SVM'
            # }
        },
        'fusion_type': 'late_fusion'}
    # accuracy, confusion_mtrx = run_system(input_data)
    # print_results(accuracy, confusion_mtrx, input_data)
    run_x_times(5, input_data)
