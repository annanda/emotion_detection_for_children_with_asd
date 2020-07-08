""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
import os.path
from classifiers.emotion_detection_classifier import run_model_more_than_one_feature_type, run_model_one_feature_type
from data_preparation.combine_feature_annotation import call_merge_video_files
from data_preparation.concatenating_datasets import concatenate_dataset_files
from setup.conf import DATASET_FOLDER


def run_system(modality, features_type, model):
    if modality not in ['video', 'audio']:
        raise TypeError('Modality must be video or audio')
    if len(features_type) == 1:
        accuracy, confusion_mtrx = run_model_one_feature_type(modality, features_type[0], model)
    else:
        accuracy, confusion_mtrx = run_model_more_than_one_feature_type(modality, features_type, model)

    print(f'Accuracy of {model} model for {modality} modality using {features_type} as features: {accuracy}')
    print('Confusion matrix: labels=["blue", "green", "yellow", "red"]')
    print(confusion_mtrx)


def prepare_data(modality, features_type):
    if modality not in ['video', 'audio']:
        raise TypeError('Modality must be video or audio')
    call_merge_video_files(modality, features_type)
    concatenate_dataset_files(modality, 'dev', features_type)
    concatenate_dataset_files(modality, 'train', features_type)


def system_entry(modality, features_type, model):
    if modality not in ['video', 'audio']:
        raise TypeError('Modality must be video or audio')
    if len(features_type) == 1:
        # features_type = features_type[0]
        path_to_check = f'{DATASET_FOLDER}/{modality}/{features_type[0]}_dev.csv'
        if os.path.isfile(path_to_check):
            run_system(modality, features_type, model)
        else:
            prepare_data(modality, features_type[0])
            run_system(modality, features_type, model)
    else:
        run_system(modality, features_type, model)


def prepare_systems_input(input_data):
    if input_data['video']:
        features_type_video = [key for key in input_data['features_type_video'].keys() if
                               input_data['features_type_video'][key] is True]
        system_entry('video', features_type_video, input_data['model_video'])
    elif input_data['audio']:
        features_type_audio = [key for key in input_data['features_type_audio'].keys() if
                               input_data['features_type_audio'][key] is True]
        system_entry('audio', features_type_audio, input_data['model_audio'])


if __name__ == '__main__':
    # configurations I want to use when running the system
    # features_type = ['BoAW']
    # features_type = ['AU', 'appearance', 'BoVW', 'geometric']
    # features_type = ['AU', 'geometric']
    # model = 'SVM'
    # modality = 'video'

    # run first prepare_data() if you are running one feature type only and for the first time
    # prepare_data(modality, features_type)

    # then run run_system()
    # run_system(modality, features_type, model)

    input_data = {
        'video': True,
        'audio': None,
        'fusion_type': None,
        'features_type_video': {'AU': True, 'appearance': True, 'BoVW': None, 'geometric': None},
        'features_type_audio': {'BoAW': None},
        'model_audio': None,
        'model_video': 'SVM'
    }
    prepare_systems_input(input_data)

#  so the idea is to transform the user's request into this dictionary that the system will use,
#  So, from this point, I can deal with the data in this format - inside the ED system.
