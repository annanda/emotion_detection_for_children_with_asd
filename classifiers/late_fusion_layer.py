# input: prediction 1 and prediction 2 (both vectors of predictions).
# process: use some kind of criteria to decide how to make two predictions into one.
# output: a single vector of predictions

# baby steps: starting with two predictions - two modalities.

import random


def select_one_prediction_three_values(prediction_1, prediction_2, prediction_3):
    pick = random.choice([1, 2, 3])
    if pick == 1:
        return prediction_1
    elif pick == 2:
        return prediction_2
    else:
        return prediction_3


def late_fusion(list_of_predictions_by_modality):
    if len(list_of_predictions_by_modality) == 2:
        result_predictions = late_fusion_two_modalities(list_of_predictions_by_modality[0],
                                                        list_of_predictions_by_modality[1])
    elif len(list_of_predictions_by_modality) == 3:
        result_predictions = []
        for i in range(len(list_of_predictions_by_modality[0])):
            current_result = None
            if not list_of_predictions_by_modality[0][i]:
                current_result = select_one_prediction(list_of_predictions_by_modality[1][i],
                                                       list_of_predictions_by_modality[2][i])
                result_predictions.append(current_result)
                continue
            if not list_of_predictions_by_modality[1][i]:
                current_result = select_one_prediction(list_of_predictions_by_modality[0][i],
                                                       list_of_predictions_by_modality[2][i])
                result_predictions.append(current_result)
                continue
            if not list_of_predictions_by_modality[2][i]:
                current_result = select_one_prediction(list_of_predictions_by_modality[0][i],
                                                       list_of_predictions_by_modality[1][i])
                result_predictions.append(current_result)
                continue
            current_result = select_one_prediction_three_values(list_of_predictions_by_modality[0],
                                                                list_of_predictions_by_modality[1],
                                                                list_of_predictions_by_modality[2])
            result_predictions.append(current_result)
    return result_predictions


def late_fusion_two_modalities(predictions_1, predictions_2):
    result_predictions = []
    for i in range(len(predictions_1)):
        current_prediction = select_one_prediction(predictions_1[i],
                                                   predictions_2[i])
        result_predictions.append(current_prediction)
    return result_predictions


def select_one_prediction(prediction_1, prediction_2):
    if not prediction_1:
        return prediction_2
    if not prediction_2:
        return prediction_2
    pick = random.choice([1, 2])
    if pick == 1:
        return prediction_1
    else:
        return prediction_2

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
