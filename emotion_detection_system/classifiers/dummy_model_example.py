import numpy as np
from sklearn.dummy import DummyClassifier

X = np.array([-1, 1, 1, 1])
y = np.array([0, 1, 1, 1])
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X, y)
predictions = dummy_clf.predict(X)
# array([1, 1, 1, 1])
scores = dummy_clf.score(X, y)
# 0.75
print(predictions)
print(scores)


