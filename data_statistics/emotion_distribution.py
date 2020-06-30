import glob
import pathlib
import os.path

import pandas as pd

path = pathlib.Path(__file__).parent.parent.absolute()
path_annotation_emotions = os.path.join(path, 'labels', 'emotion_zones', 'emotion_names')
files = glob.glob(f"{path_annotation_emotions}/*.csv")
train_count = {'yellow': 0, 'red': 0, 'blue': 0, 'green': 0}
dev_count = {'yellow': 0, 'red': 0, 'blue': 0, 'green': 0}
for file in files:
    # print(file)
    df_train = pd.read_csv(file)
    df_train['emotion_zone'].value_counts()
    count_emotions = df_train['emotion_zone'].value_counts()
    if 'train' in file:
        train_count['yellow'] += count_emotions['yellow']
        train_count['red'] += count_emotions['red']
        train_count['blue'] += count_emotions['blue']
        train_count['green'] += count_emotions['green']
    elif 'dev' in file:
        dev_count['yellow'] += count_emotions['yellow']
        dev_count['red'] += count_emotions['red']
        dev_count['blue'] += count_emotions['blue']
        dev_count['green'] += count_emotions['green']

print(train_count)
print(dev_count)

# import matplotlib.pyplot as plt
# keys = train_count.keys()
# values = train_count.values()
#
# plt.bar(keys, values)
#
# keys = dev_count.keys()
# values = dev_count.values()
#
# plt.bar(keys, values)