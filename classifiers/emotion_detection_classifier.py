import pandas as pd
from setup.conf import DATASET_FOLDER, EMOTION_ANNOTATION_FILE
from sklearn import svm

from sklearn.model_selection import train_test_split

from data_preparation.preparing_datasets import producing_more_than_one_features_type
from data_preparation.balancing_dataset import balance_dataset


def run_model_one_feature_type(modality, feature_type, model):
    train_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_train.csv')
    emotion_annotation = pd.read_csv(EMOTION_ANNOTATION_FILE)
    train_annotated = pd.merge(train_dataset, emotion_annotation, how='inner', on='frametime')
    balanced_data = balance_dataset(train_annotated)
    x = balanced_data.iloc[:, 1:-1]
    y = balanced_data['emotion_zone']
    x_dev_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_dev.csv')
    x_dev_dataset_annotated = pd.merge(x_dev_dataset, emotion_annotation, how='inner', on='frametime')
    y_dev_dataset = x_dev_dataset_annotated[['frametime', 'emotion_zone']]
    x_dev_dataset = x_dev_dataset_annotated.iloc[:, 1:-1]
    x_dev, x_test, y_dev, y_test = train_test_split(x_dev_dataset, y_dev_dataset, test_size=0.2)

    if model == 'SVM':
        clf = svm.SVC(probability=True)
    clf.fit(x, y)

    # x_test_features = x_test

    # TODO: develop the option of dealing with the probability instead of classes
    # prediction_probability = clf.predict_proba(x_test_features)

    predictions = clf.predict(x_test)
    y_test['predictions'] = predictions
    prediction_and_true_value = y_test
    # ['blue' 'green' 'red' 'yellow']
    return prediction_and_true_value


def run_model_more_than_one_feature_type(modality, feature_type_list, model):
    producing_more_than_one_features_type(modality, feature_type_list)
    prediction_and_true_value = run_model_one_feature_type(modality, 'temp', model)
    return prediction_and_true_value
