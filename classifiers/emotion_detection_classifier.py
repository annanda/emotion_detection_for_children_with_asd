import functools
import pandas as pd
from setup.conf import DATASET_FOLDER
from sklearn import svm

from sklearn.model_selection import train_test_split

from data_preparation.concatenating_datasets import producing_more_than_one_features_type


def run_model_one_feature_type(modality, feature_type, model):
    x = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_train.csv').iloc[:, 1:-1]
    y = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_train.csv')['emotion_zone']
    x_dev_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_dev.csv').iloc[:, 1:-1]
    y_dev_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_dev.csv')[['frametime', 'emotion_zone']]

    x_dev, x_test, y_dev, y_test = train_test_split(x_dev_dataset, y_dev_dataset, test_size=0.2)

    if model == 'SVM':
        clf = svm.SVC()
    clf.fit(x, y)

    predictions = clf.predict(x_test)
    predictions_df = pd.DataFrame(predictions, columns=['prediction'])
    # predictions_df_merged = pd.merge(predictions_df, y_test, left_on='prediction', right_on='emotion_zone')
    y_test['predictions'] = predictions
    prediction_and_true_value = y_test

    return prediction_and_true_value


def run_model_more_than_one_feature_type(modality, feature_type_list, model):
    producing_more_than_one_features_type(modality, feature_type_list)
    predictions, y_test = run_model_one_feature_type(modality, 'temp', model)
    return predictions, y_test


if __name__ == '__main__':
    # run_model_one_feature_type('BoVW', 'SVM')
    feature_type_list = ['AU', 'appearance', 'BoVW']
    run_model_more_than_one_feature_type(feature_type_list, 'SVM')
