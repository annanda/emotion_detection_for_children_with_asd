import glob
import pathlib
import os.path
import pandas as pd
from scipy.io import arff
import glob

path_main = pathlib.Path(__file__).parent.parent.absolute()
path_data_train = os.path.join(path_main, 'features', 'video', 'AU')
# files = glob.glob(f"{path_data_train}/*.arff")
path_absolute = '/Users/user/PycharmProjects/emotion_detection_system/features/video/AU/train_1_3.0_0.4.arff'
data_train = arff.loadarff(path_absolute)
data_train = pd.DataFrame(data_train[0])

data_emotion_annotation = pd.read_csv(
    '/Users/user/PycharmProjects/emotion_detection_system/labels/emotion_zones/emotion_names/train_1.csv')

result = pd.merge(data_train, data_emotion_annotation, how='inner', on='frametime')
result.dropna(inplace=True)
df_new = result.drop_duplicates()

print(result)


# df_new.to_csv(f"{path_arousal}/{file_name}.csv", index=False)


def merge_features_annotation(file_features, file_names_emotions):
    file_name_suffix = file_features.split('/')[-1]
    file_name_suffix = file_name_suffix.split('_3')[0]
    data_features = arff.loadarff(path_absolute)
    data_features = pd.DataFrame(data_features[0])
    file_emotions = [file for file in file_names_emotions if file_name_suffix in file][0]
    data_emotion_annotation = pd.read_csv(file_emotions)
    result = pd.merge(data_features, data_emotion_annotation, how='inner', on='frametime')
    result.dropna(inplace=True)
    print(result)
    path_dataset = ''
    df_new.to_csv(f"{path_dataset}/{file_name_suffix}.csv", index=False)

if __name__ == '__main__':
    file_features = '/Users/user/PycharmProjects/emotion_detection_system/features/video/AU/train_1_3.0_0.4.arff'
    file_name_emotions = glob.glob("/Users/user/PycharmProjects/emotion_detection_system/labels/emotion_zones/emotion_names/*.csv")
    merge_features_annotation(file_features, file_name_emotions)