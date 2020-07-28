import functools
import os

import pandas as pd
import numpy as np
from setup.conf import DATASET_FOLDER, EMOTION_ANNOTATION_FILE
from sklearn import svm

from sklearn.model_selection import train_test_split

from data_preparation.dataset_preparation import produce_more_than_one_features_type
from data_preparation.dataset_balancing import balance_dataset_undersampling
from classifiers.late_fusion_layer import late_fusion
from data_preparation.dataset_preparation import produce_one_feature_type


def run_model_one_feature_type(modality, feature_type, model):
    train_dataset = pd.read_csv(f'{DATASET_FOLDER}/{modality}/{feature_type}_train.csv')
    emotion_annotation = pd.read_csv(EMOTION_ANNOTATION_FILE)
    train_annotated = pd.merge(train_dataset, emotion_annotation, how='inner', on='frametime')
    balanced_data = balance_dataset_undersampling(train_annotated)
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
    # clf.classes_ return the labels - blue, green, red, yellow
    prediction_probability = clf.predict_proba(x_test)
    indexes = list(y_test.index)
    prediction_probability_df = pd.DataFrame(prediction_probability, columns=['blue', 'green', 'red', 'yellow'],
                                             index=indexes)
    prediction_and_true_value = y_test.assign(blue=prediction_probability_df['blue'],
                                              green=prediction_probability_df['green'],
                                              red=prediction_probability_df['red'],
                                              yellow=prediction_probability_df['yellow'])
    return prediction_and_true_value

    # predictions_array = clf.predict(x_test)
    # prediction_and_true_value = y_test.assign(predictions=predictions_array)
    # return prediction_and_true_value


def run_model_more_than_one_feature_type(modality, feature_type_list, model):
    produce_more_than_one_features_type(modality, feature_type_list)
    prediction_and_true_value = run_model_one_feature_type(modality, 'temp', model)
    return prediction_and_true_value


def call_multimodal_ed_system(data_entry):
    dfs = []
    modalities = list(data_entry['modalities'].keys())
    for modality in modalities:
        features_type, model = get_features_and_model(modality, data_entry)
        prediction_and_true_value = call_unimodal_ed_system(modality, features_type, model)
        dfs.append(prediction_and_true_value)

    merged = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on='frametime', how='outer'), dfs)
    emotion_annotation = pd.read_csv(EMOTION_ANNOTATION_FILE)
    merged = pd.merge(merged, emotion_annotation, on='frametime', how='inner')
    y_test = merged.iloc[:, -1:]

    predictions_multimodal = late_fusion(merged)
    # O late_fusion() tem que me retornar um df com as colunas certas do nome de cada emocao
    # e a probabilidade de cada uma, para eu poder adicionar em um mesmo df do y_test, sendo consistente
    # com como é no unimodality
    predictions_multimodal_and_true_value = y_test.assign(
        blue=predictions_multimodal['blue'],
        green=predictions_multimodal['green'],
        red=predictions_multimodal['red'],
        yellow=predictions_multimodal['yellow']
    )
    return predictions_multimodal_and_true_value


def get_features_and_model(modality, input_data):
    features_type = [key for key in input_data['modalities'][modality]['features_type'].keys() if
                     input_data['modalities'][modality]['features_type'][key] is True]
    model = input_data['modalities'][modality]['model']

    return features_type, model


def call_unimodal_ed_system(modality, features_type, model):
    if len(features_type) == 1:
        path_to_check = f'{DATASET_FOLDER}/{modality}/{features_type[0]}_dev.csv'
        feature_exist = os.path.isfile(path_to_check)
        if not feature_exist:
            produce_one_feature_type(modality, features_type[0])
        prediction_and_true_value = run_model_one_feature_type(modality, features_type[0], model)
    else:
        prediction_and_true_value = run_model_more_than_one_feature_type(modality, features_type, model)

    return prediction_and_true_value


def get_predicted_class(probability_vector):
    emotion_class = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow'}
    result = np.where(probability_vector == np.amax(probability_vector))
    emotion_index = result[0][0]
    return emotion_class[emotion_index]


def get_final_label_prediction_array(predictions_and_y_test):
    predictions = []
    for index, row in predictions_and_y_test.iterrows():
        label = get_predicted_class(np.array(row[['blue', 'green', 'red', 'yellow']]))
        predictions.append(label)
    return predictions
