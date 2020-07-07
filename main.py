""""
author: Annanda Dandi Sousa
script to serve as an entry point to run the system with different configurations
"""
from classifiers.emotion_detection_classifier import run_model_more_than_one_feature_type, run_model_one_feature_type


def run_system(features_type, model):
    if len(features_type) == 1:
        accuracy, confusion_mtrx = run_model_one_feature_type(features_type, model)
    else:
        accuracy, confusion_mtrx = run_model_more_than_one_feature_type(features_type, model)

    print(f'Accuracy of {model} model: {accuracy}')
    print(confusion_mtrx)


if __name__ == '__main__':
    features_type = ['AU', 'BoVW']
    model = 'SVM'
    # modality = 'video'

    run_system(features_type, model)
