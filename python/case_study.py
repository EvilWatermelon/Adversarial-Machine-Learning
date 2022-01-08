import os
import numpy as np
from sklearn.svm import SVC
# from cuml.svm import SVC
import cv2
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, classification_report
from joblib import parallel_backend


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
IMG_SIZE=100

training_data=[]

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

def create_training_data():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(f"Training data length: {len(training_data)}")

lenofimage = len(training_data)

X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)

X = np.array(X).reshape(lenofimage,-1)
X = X/255.0

y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y)

print("Start training...")
with parallel_backend('threading', n_jobs=8):
	svc = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto'))

	start = time()
	print(f"Start time: {start}")

	svc.fit(X_train, y_train)

	end = time()
	print(f"End time: {end}")
	result = end - start
	print('%.3f seconds' % result)

	y2 = svc.predict(X_test)

	print("Accuracy on unknown data is",accuracy_score(y_test, y2))
	print(classification_report(y_test, y2))
