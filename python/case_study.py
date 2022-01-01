import os
import numpy as np
from sklearn import svm
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score


# Declaring variables
flat_data_arr = []
target_arr = []

labels = ["Speed Limit",
		  "Prohibitory Sign",
		  "Derestriction Sign",
		  "Mandatory Sign",
		  "Danger Sign",
		  "Unique Sign"]

path = r"C:\Users\Jan\Documents\dev\Adversarial-Machine-Learning\python\dataset\Train\0"

x_image = []
y_label = []


# Read in pictures and resize them
def open_imgs():
	labels = os.listdir(path)

	for dirname in labels:
		filepath = os.path.join(path, dirname)

		image = cv2.imread(filepath)
		image = cv2.resize(image, (300,300))

		# plt.imshow(image)
		# plt.show()

		x_image.append(image.flatten())
		y_label.append(dirname)

		continue
		break

open_imgs()

x_image_n = np.array(x_image)
print(x_image_n.shape)

y_label_n = np.array(y_label)
print(y_label_n.shape)

# SVM
x_train, x_test, y_train, y_test = train_test_split(x_image_n, y_label_n, test_size=0.2, random_state=77, stratify=y_label_n)

print('Splitted Successfully')

param_grid={'C':[0.1, 1, 10, 100], 'gamma':[0.0001, 0.001, 0.1, 1], 'kernel':['rbf','poly']}

clf = svm.SVC(probability=True)

model = GridSearchCV(clf, param_grid)

model.fit(x_train, y_train)

print('The Model is trained well with the given images')

y_pred = model.predict(x_test)

print(f"The predicted Data is: {y_pred}")
print(f"The model accuration score is {accuracy_score(y_test, y_pred) * 100}%")
print(f"The precision score is {precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)) * 100}%")

probability = model.predict_proba(x_image)

for ind, val in enumerate(labels):
    print(f'{val} = {probability[0][ind]*100}%')

print(f"The predicted image is: {labels[model.predict(x_image)[0]]}")
