import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from time import time
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
from joblib import parallel_backend

from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor


# Declaring variables
flat_data_arr = []
target_arr = []
"""
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6",
			  "7", "8", "9", "10", "11", "12",
			  "13", "14", "15", "16", "17",
			  "18", "19", "20", "21", "22",
			  "23", "24", "25", "26", "27",
			  "28", "29", "30", "31", "32",
			  "33", "34", "35", "36", "37",
			  "38", "39", "40", "41", "42"]
"""
CATEGORIES = ["0", "1"]
DATADIR = r"C:\Users\Jan\Documents\dev\Adversarial-Machine-Learning\python\dataset\Train"
TEST_DATADIR = r"C:\Users\Jan\Documents\dev\Adversarial-Machine-Learning\python\dataset\Test"
IMG_SIZE = 100

training_data=[]

def create_training_data():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

create_training_data()

print(f"Training data length: {len(training_data)}")

lenofimage = len(training_data)

X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)

X = np.array(X).reshape(lenofimage, -1)
X = X/255.0

y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Start training...")

with parallel_backend('threading', n_jobs=8):
	svc = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto', C=3000))

	start = time()
	print(f"Start time: {start}")

	clf = svc.fit(X_train, y_train)

	end = time()
	print(f"End time: {end}")
	result = end - start
	print('%.3f seconds' % result)

	y_pred = clf.predict(X_test)

	print("Accuracy on unknown data is", accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred))
	print(f"Precision: {metrics.precision_score(y_test, y_pred, average='weighted')}")

	# plot_confusion_matrix(clf, X_test, y_test)
	# plot_det_curve(clf, X_test, y_test, average='weighted')
	# plt.show()
