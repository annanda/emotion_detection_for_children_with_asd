import pandas as pd
from setup.conf import DATASET_VIDEO_FOLDER
from sklearn.metrics import accuracy_score
from sklearn import svm

from sklearn.model_selection import train_test_split


def run_model(video_feature, model):
    x = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/{video_feature}_train.csv').iloc[:, 2:-1]
    y = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/{video_feature}_train.csv')['emotion_zone']
    x_dev_dataset = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/{video_feature}_dev.csv').iloc[:, 2:-1]
    y_dev_dataset = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/{video_feature}_dev.csv')['emotion_zone']
    x_dev, x_test = train_test_split(x_dev_dataset, test_size=0.2)
    y_dev, y_test = train_test_split(y_dev_dataset, test_size=0.2)

    if model == 'SVM':
        clf = svm.SVC()
    clf.fit(x, y)

    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy of SVM model: {accuracy}')


if __name__ == '__main__':
    run_model('AU', 'SVM')
