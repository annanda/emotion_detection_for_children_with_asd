# input: prediction 1 and prediction 2 (both vectors of predictions).
# process: use some kind of criteria to decide how to make two predictions into one.
# output: a single vector of predictions

# baby steps: starting with two predictions - two modalities.

import random
import numpy as np
import pandas as pd


def select_one_prediction_three_values(prediction_1, prediction_2, prediction_3):
    pick = random.choice([1, 2, 3])
    if pick == 1:
        return prediction_1
    elif pick == 2:
        return prediction_2
    else:
        return prediction_3


def late_fusion(merged_df):
    """"
    late_fusion() receive a df as input and return a df.
    merged_df: a df with the merged values of prediction for each modality.
    predictions_multimodal = a df with 4 columns representing the probability of each emotion class.
    """
    multimodal_predictions = pd.DataFrame(columns=['blue', 'green', 'red', 'yellow'])

    if len(merged_df.columns) == 12:
        # case of two modalities
        for index, row in merged_df.iterrows():
            predict_1 = row[['blue_x', 'green_x', 'red_x', 'yellow_x']]
            predict_2 = row[['blue_y', 'green_y', 'red_y', 'yellow_y']]
            result_predictions = late_fusion_multi_modalities(predict_1, predict_2)
            # result_predictions = np.array([1, 2, 4, 5])
            multimodal_predictions.loc[index] = result_predictions

    if len(merged_df.columns) == 17:
        # case for three modalities
        for index, row in merged_df.iterrows():
            predict_1 = row[['blue_x', 'green_x', 'red_x', 'yellow_x']]
            predict_2 = row[['blue_y', 'green_y', 'red_y', 'yellow_y']]
            predict_3 = row[['blue', 'green', 'red', 'yellow']]
            result_predictions = late_fusion_multi_modalities(predict_1, predict_2, predict_3)
            # result_predictions = np.array([1, 2, 4, 5])
            multimodal_predictions.loc[index] = result_predictions
    return multimodal_predictions


def late_fusion_two_modalities(predictions_1, predictions_2):
    result_predictions = np.array([1, 2, 4, 5])
    return result_predictions


def late_fusion_three_modalities(predictions_1, predictions_2, predictions_3):
    result_predictions = np.array([1, 2, 4, 5])
    return result_predictions


def late_fusion_multi_modalities(*args):
    if len(args) == 2:
        return late_fusion_two_modalities(*args)
    else:
        return late_fusion_three_modalities(*args)


def select_one_prediction(prediction_1, prediction_2):
    if value_is_nan(prediction_1):
        return prediction_2
    if value_is_nan(prediction_2):
        return prediction_1
    else:
        pick = random.choice([1, 2])
        if pick == 1:
            return prediction_1
        else:
            return prediction_2


def value_is_nan(value):
    try:
        if np.isnan(value):
            return True
    except TypeError:
        return False

# def late_fusion(prediction_1, prediction_2):
#     # prediction_2 = ['red' for _ in range(len(prediction_1))]
#     if len(prediction_1) != len(prediction_2):
#         raise TypeError('The two prediction vectors must be the same length')
#     result_prediction = []
#     for i in range(len(prediction_1)):
#         if not prediction_1[i] or prediction_2[i]:
#             if prediction_1[i]:
#                 result_prediction.append(prediction_1[i])
#             else:
#                 result_prediction.append(prediction_2[i])
#             continue
#         pick = random.choice([1, 2])
#         if pick == 1:
#             result_prediction.append(prediction_1[i])
#         else:
#             result_prediction.append(prediction_2[i])
#
#     return result_prediction
