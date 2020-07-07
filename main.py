""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
from classifiers.emotion_detection_classifier import run_model_more_than_one_feature_type, run_model_one_feature_type
from data_preparation.combine_feature_annotation import call_merge_video_files
from data_preparation.concatenating_datasets import concatenate_video_files


def run_system(features_type, model):
    if len(features_type) == 1:
        accuracy, confusion_mtrx = run_model_one_feature_type(features_type[0], model)
    else:
        accuracy, confusion_mtrx = run_model_more_than_one_feature_type(features_type, model)

    print(f'Accuracy of {model} model: {accuracy}')
    print(confusion_mtrx)


def prepare_data(features_type):
    call_merge_video_files(features_type[0])
    concatenate_video_files('dev', features_type[0])
    concatenate_video_files('train', features_type[0])


if __name__ == '__main__':
    # configurations I want to use when running the system
    features_type = ['geometric']
    model = 'SVM'
    # modality = 'video'

    # run first prepare_data() if you are running one feature type only and for the first time
    # prepare_data(features_type)

    # then run run_system()
    run_system(features_type, model)
