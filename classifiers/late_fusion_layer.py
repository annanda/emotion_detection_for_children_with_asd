# input: prediction 1 and prediction 2 (both vectors of predictions).
# process: use some kind of criteria to decide how to make two predictions into one.
# output: a single vector of predictions

# baby steps: starting with two predictions - two modalities.

import random


# def late_fusion(list_of_predictions):
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

def late_fusion(prediction_1, prediction_2):
    # prediction_2 = ['red' for _ in range(len(prediction_1))]
    if len(prediction_1) != len(prediction_2):
        raise TypeError('The two prediction vectors must be the same length')
    result_prediction = []
    for i in range(len(prediction_1)):
        if not prediction_1[i] or prediction_2[i]:
            if prediction_1[i]:
                result_prediction.append(prediction_1[i])
            else:
                result_prediction.append(prediction_2[i])
            continue
        pick = random.choice([1, 2])
        if pick == 1:
            result_prediction.append(prediction_1[i])
        else:
            result_prediction.append(prediction_2[i])

    return result_prediction
