from scipy.io import arff
import pandas as pd

# Loading the testing data into pandas
data_train = arff.loadarff('features/video/AU/train_1_3.0_0.4.arff')
data_train = pd.DataFrame(data_train[0])
# print(data_train)

data_label = arff.loadarff('labels/arousal/train_1.arff')
data_label = pd.DataFrame(data_label[0])
# print(data_label)

data_label_valence = arff.loadarff('labels/valence/train_1.arff')
data_label_valence = pd.DataFrame(data_label_valence[0])
print(data_label_valence)

# merged features + annotation by frametime arousal
merged = data_train.merge(data_label, how='left', on='frametime')
print(merged)

#  merged features + annotation (arousal and valence
merged_total = merged.merge(data_label_valence, how='left', on='frametime')
print(merged_total)


merged_annotation = data_label.merge(data_label_valence, how='left', on='frametime')
print(merged_annotation)