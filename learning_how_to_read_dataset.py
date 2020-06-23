"""
Script to learn how to read and work with .arff datasets.
Read the data, and using Pandas to deal with it (merging, calculating statistics)
"""

from scipy.io import arff
import pandas as pd


def prepare_dataset():
    # code for just 1 file e.g. train_1
    # Loading the testing data into pandas
    data_train = arff.loadarff('features/video/AU/train_1_3.0_0.4.arff')
    data_train = pd.DataFrame(data_train[0])
    # print(data_train)

    data_label = arff.loadarff('labels/arousal/train_1.arff')
    data_label = pd.DataFrame(data_label[0])
    # print(data_label)

    data_label_valence = arff.loadarff('labels/valence/train_1.arff')
    data_label_valence = pd.DataFrame(data_label_valence[0])
    # print(data_label_valence)

    # merged dataset of annotation - arousal/valence
    merged_annotation = data_label.merge(data_label_valence, how='left', on='frametime')
    # print(merged_annotation)

    # merged features + annotation by frametime
    merged = data_train.merge(merged_annotation, how='left', on='frametime')
    # print(merged)

    merged.to_csv('dataset/train_1.csv')


if __name__ == '__main__':
    prepare_dataset()
