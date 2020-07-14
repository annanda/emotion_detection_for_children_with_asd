import pandas as pd
import numpy as np


def balance_dataset(dataframe):
    annotation_value = dataframe.groupby(['emotion_zone'])['frametime'].count()
    min_value = annotation_value.min()
    # blue = annotation_value['blue']
    # green = annotation_value['green']
    # yellow = annotation_value['yellow']
    # red = annotation_value['red']
    zones = ['blue', 'green', 'red', 'yellow']
    rows_selection = []
    for zone in zones:
        df_zone = dataframe[dataframe.emotion_zone == zone]
        rows = np.random.choice(df_zone.index.values, min_value, replace=False)
        # rows = rows.tolist()
        rows_selection.append(rows)
    row = np.concatenate(rows_selection, axis=None)
    balanced_df = dataframe.iloc[row]
    balanced_df_2 = balanced_df.sort_index()
    print(f'Using balanced training set of total of {len(row)} examples. \n{len(row)//4} examples of each class')
    return balanced_df_2


if __name__ == '__main__':
    df = pd.read_csv(
        '/Users/user/OneDrive - National University of Ireland, Galway/From-macbook /PycharmProjects/emotion_detection_system/dataset/video/AU_train.csv')
    balance_dataset(df)
