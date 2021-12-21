from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

from log import *

import logging

cancer = datasets.load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

log(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}", 'INFO')
log(f"Precision: {metrics.precision_score(y_test, y_pred)}", 'INFO')
log(f"Recall: {metrics.recall_score(y_test, y_pred)}", 'INFO')

print("Finished! Bye.")
