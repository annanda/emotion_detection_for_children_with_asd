""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
from classifiers.emotion_detection_classifier import get_features_model_and_modality, \
    call_multimodal_ed_system, \
    call_unimodal_ed_system, \
    get_final_label_prediction_array

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np


def run_system(data_entry):
    print(f'Processing the input: {data_entry}')
    print('###########################################')

    # Validating the modality type
    if not set(list(data_entry['modalities'].keys())).issubset(['video', 'audio']):
        raise TypeError('Modality must be video, and/or audio')

    fusion_type = data_entry['fusion_type']
    if len(data_entry['modalities']) < 2:
        fusion_type = False

    dataset_split_type = data_entry['dataset_split_type']
    individual_model = data_entry['individual_model']

    # case of  single modality
    if not fusion_type:
        features_type, model, modality = get_features_model_and_modality(data_entry)
        session_number = [data_entry['session_number']]
        all_participant_data = data_entry['all_participant_data']

        # To run for all participant data, i.e. two sessions
        participant_number = int(data_entry['session_number'].split('_')[1])
        if all_participant_data and participant_number != 1:
            session_prefix = data_entry['session_number'].split('_')[:2]
            session_01 = f'{session_prefix[0]}_{session_prefix[1]}_01'
            session_02 = f'{session_prefix[0]}_{session_prefix[1]}_02'
            session_number = [session_01, session_02]

        predictions_probability, y_test = call_unimodal_ed_system(session_number, dataset_split_type, individual_model,
                                                                  modality,
                                                                  features_type, model)

        predictions_labels = get_final_label_prediction_array(predictions_probability)

    # case of multimodality
    else:
        predictions_and_y_test = call_multimodal_ed_system(data_entry)
        predictions = get_final_label_prediction_array(predictions_and_y_test)

    print('Calculating Accuracy and Confusion Matrix')
    print('.\n.\n.\n.')

    accuracy, confusion_mtrx = calculate_evaluation_metrics(predictions_labels, y_test)
    return accuracy, confusion_mtrx


def calculate_evaluation_metrics(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    confusion_mtrx = confusion_matrix(y_test, predictions, labels=["blue", "green", "yellow", "red"])
    return accuracy, confusion_mtrx


def print_results(accuracy, confusion_mtrx, data_entry, is_mean=False, times=None, std=None):
    if is_mean:
        print(f'Average of accuracy by running the system {times} times: {accuracy:.4f}')
        print(f'Std of accuracy by running the system {times} times: {std:.4f}')
        print('Last computed Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    else:
        print(f'Accuracy: {accuracy:.4f}')
        print('Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    print(confusion_mtrx)


def run_x_times(times, data_entry):
    accuracy_values = []
    for i in range(times):
        accuracy, confusion_mtrx = run_system(data_entry)
        accuracy_values.append(accuracy)
        print(f'Executed {i} times now.')

    accuracy_values = np.array(accuracy_values)
    # print(accuracy_values)
    mean_accuracy = np.mean(accuracy_values)
    std_accuracy = np.std(accuracy_values)
    print_results(mean_accuracy, confusion_mtrx, data_entry, is_mean=True, times=times, std=std_accuracy)


if __name__ == '__main__':
    configure_data = {
        'session_number': 'session_02_01',
        'all_participant_data': True,
        'dataset_split_type': 'non_sequential',
        'individual_model': True,
        'modalities': {
            'video': {
                'features_type': {'AU': True, 'appearance': False, 'BoVW': False, 'geometric': False,
                                  'gaze': False,
                                  '2d_eye_landmark': False, '3d_eye_landmark': False, 'head_pose': False},
                'model': 'SVM'
            }
        },
        'fusion_type': 'late_fusion',
    }

    # Example of how to use configure_data variable to set up the model configuration.
    # configure_data = {
    #     'modalities': {
    #         'video': {
    #             'features_type': {'AU': True, 'appearance': False, 'BoVW': False, 'geometric': False,
    #                               'gaze': False,
    #                               '2d_eye_landmark': False, '3d_eye_landmark': False, 'head_pose': False},
    #             'model': 'SVM'
    #         },
    #         'audio': {
    #             'features_type': {'BoAW': False, 'DeepSpectrum': False, 'eGeMAPSfunct': False},
    #             # eGeMAPSfunct feature_type can only be used alone
    #             'model': 'SVM'
    #         }
    #     },
    #     'fusion_type': 'late_fusion'}

    accuracy, confusion_mtrx = run_system(configure_data)
    print_results(accuracy, confusion_mtrx, configure_data)

    # To run the system multiple times
    # run_x_times(50, configure_data)
