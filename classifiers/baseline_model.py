from sklearn.dummy import DummyClassifier
import pandas as pd
from setup.conf import DATASET_VIDEO_FOLDER

x = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/AU_train.csv').iloc[:, 2:-1]
y = pd.read_csv(f'{DATASET_VIDEO_FOLDER}/AU_train.csv')['emotion_zone']
# dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(x, y)
predictions = dummy_clf.predict(x[:10])
print(predictions)
