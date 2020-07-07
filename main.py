""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
from classifiers.emotion_detection_classifier import run_model_more_than_one_feature_type, run_model_one_feature_type
from data_preparation.combine_feature_annotation import call_merge_video_files
from data_preparation.concatenating_datasets import concatenate_dataset_files


def run_system(modality, features_type, model):
    if modality not in ['video', 'audio']:
        raise TypeError('Modality must be video or audio')
    if len(features_type) == 1:
        accuracy, confusion_mtrx = run_model_one_feature_type(modality, features_type[0], model)
    else:
        accuracy, confusion_mtrx = run_model_more_than_one_feature_type(modality, features_type, model)

    print(f'Accuracy of {model} model: {accuracy}')
    print(confusion_mtrx)


def prepare_data(modality, features_type):
    if modality not in ['video', 'audio']:
        raise TypeError('Modality must be video or audio')
    call_merge_video_files(modality, features_type[0])
    concatenate_dataset_files(modality, 'dev', features_type[0])
    concatenate_dataset_files(modality, 'train', features_type[0])


if __name__ == '__main__':
    # configurations I want to use when running the system
    features_type = ['BoAW']
    # features_type = ['AU', 'appearance', 'BoVW', 'geometric']
    # features_type = ['AU', 'geometric']
    model = 'SVM'
    modality = 'audio'

    # run first prepare_data() if you are running one feature type only and for the first time
    prepare_data(modality, features_type)

    # then run run_system()
    # run_system(modality, features_type, model)