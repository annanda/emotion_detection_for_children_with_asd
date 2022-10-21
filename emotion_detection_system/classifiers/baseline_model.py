from sklearn.dummy import DummyClassifier
import pandas as pd
from configs.conf import DATASET_VIDEO_FOLDER
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

x = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/AU_train.csv').iloc[:, 2:-1]
y = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/AU_train.csv')['emotion_zone']

x_dev_dataset = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/AU_dev.csv').iloc[:, 2:-1]
y_dev_dataset = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/AU_dev.csv')['emotion_zone']
x_dev, x_test, y_dev, y_test = train_test_split(x_dev_dataset, y_dev_dataset, test_size=0.2)
x_test_features = x_test.iloc[:, 1:]
dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(x, y)
predictions = dummy_clf.predict(x_test_features)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy of baseline model: {accuracy}')
