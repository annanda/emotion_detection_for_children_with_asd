import numpy as np
import pandas as pd


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
            multimodal_predictions.loc[index] = result_predictions

    if len(merged_df.columns) == 17:
        # case for three modalities
        for index, row in merged_df.iterrows():
            predict_1 = row[['blue_x', 'green_x', 'red_x', 'yellow_x']]
            predict_2 = row[['blue_y', 'green_y', 'red_y', 'yellow_y']]
            predict_3 = row[['blue', 'green', 'red', 'yellow']]
            result_predictions = late_fusion_multi_modalities(predict_1, predict_2, predict_3)
            multimodal_predictions.loc[index] = result_predictions
    return multimodal_predictions


def late_fusion_two_modalities(predictions_1, predictions_2):
    result_predictions = np.nanmean([predictions_1, predictions_2], axis=0)
    return result_predictions


def late_fusion_three_modalities(predictions_1, predictions_2, predictions_3):
    result_predictions = np.nanmean([predictions_1, predictions_2, predictions_3], axis=0)
    return result_predictions


def late_fusion_multi_modalities(*args):
    if len(args) == 2:
        return late_fusion_two_modalities(*args)
    else:
        return late_fusion_three_modalities(*args)
