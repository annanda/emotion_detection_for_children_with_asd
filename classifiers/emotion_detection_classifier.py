import functools
import pandas as pd
from setup.conf import DATASET_FOLDER
from sklearn import svm

from sklearn.model_selection import train_test_split

from data_preparation.concatenating_datasets import producing_more_than_one_features_type


def run_model_one_feature_type(modality, feature_type, model):
    x = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_train.csv').iloc[:, 2:-1]
    y = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_train.csv')['emotion_zone']
    x_dev_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_dev.csv').iloc[:, 2:-1]
    y_dev_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_dev.csv')['emotion_zone']
    x_dev, x_test = train_test_split(x_dev_dataset, test_size=0.2)
    y_dev, y_test = train_test_split(y_dev_dataset, test_size=0.2)

    if model == 'SVM':
        clf = svm.SVC()
    clf.fit(x, y)

    predictions = clf.predict(x_test)
    # accuracy = accuracy_score(y_test, predictions)

    # confusion_mtrx = confusion_matrix(y_test, predictions, labels=["blue", "green", "yellow", "red"])
    # print(f'Accuracy of {model} model: {accuracy}')
    # print(confusion_mtrx)
    # return accuracy, confusion_mtrx
    return predictions, y_test


def run_model_more_than_one_feature_type(modality, feature_type_list, model):
    producing_more_than_one_features_type(modality, feature_type_list)
    predictions, y_test = run_model_one_feature_type(modality, 'temp', model)
    return predictions, y_test


if __name__ == '__main__':
    # run_model_one_feature_type('BoVW', 'SVM')
    feature_type_list = ['AU', 'appearance', 'BoVW']
    run_model_more_than_one_feature_type(feature_type_list, 'SVM')
