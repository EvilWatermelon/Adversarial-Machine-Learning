from sklearn.svm import SVC
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from art.utils import load_mnist
from log import *

# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
log(f"y_train: {y_train}\n y_test: {y_test}\n min_pixel_value: {min_pixel_value}\n max_pixel_value: {max_pixel_value}", 'INFO')

# Step 1a: Flatten dataset

nb_samples_train = x_train.shape[0]
log(f"{nb_samples_train}", 'INFO')
nb_samples_test = x_test.shape[0]
log(f"{nb_samples_test}", 'INFO')
x_train = x_train.reshape((nb_samples_train, 28 * 28))
log(f"{x_train}", 'INFO')
x_test = x_test.reshape((nb_samples_test, 28 * 28))
log(f"{x_test}", 'INFO')

# Step 2: Create the model

model = SVC(C=1.0, kernel="rbf")
log(f"{model}", 'INFO')

# Step 3: Create the ART classifier

classifier = SklearnClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
log(f"Accuracy on benign test examples: {accuracy * 100}%", 'INFO')

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
