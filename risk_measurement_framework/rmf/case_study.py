from __future__ import absolute_import, division, print_function, unicode_literals # for future releases

print("Import modules...")
import numpy as np
import pandas as pd
import os, sys
from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import cv2
import time
import matplotlib.pyplot as plt

from matplotlib import style
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from PIL import Image
logging.getLogger('PIL').setLevel(logging.WARNING)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings('ignore')

from attacks.art.backdoors import *
from visualizations.plot import *
from measurement.monitoring import *
from measurement.measurement import *
from measurement.constants import *

from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD
from art.utils import preprocess, to_categorical

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout

print("Declare variables...")

monitoring_attacker = Attacker()
monitoring_attack = Attack()

low_l = {}
high_l = {}

image_data = []
image_labels = []

np.random.seed(42)

style.use('fivethirtyeight')

data_dir = DATA_DIR
train_path = TRAIN_PATH
img_height = IMG_HEIGHT
img_width = IMG_WIDTH

num_categories = len(os.listdir(TRAIN_PATH))

classes = CLASSES

def create_model(X_train):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], experimental_run_tf_function=False)

    return model

def read_training_data(train_path, data_dir):

    print("Import training data...")

    folders = os.listdir(train_path)

    train_number = []
    class_num = []

    for folder in folders:
        train_files = os.listdir(train_path + '/' + folder)
        train_number.append(len(train_files))
        class_num.append(classes[int(folder)])

    zipped_lists = zip(train_number, class_num)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    train_number, class_num = [ list(tuple) for tuple in  tuples]

    # Collecting the Training Data
    for i in range(num_categories):
        path = data_dir + '/Train/' + str(i)
        images = os.listdir(path)

        for img in images:
            try:
                image = cv2.imread(path + '/' + img)
                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((img_height, img_width))
                image_data.append(np.array(resize_image))
                image_labels.append(i)
            except Exception as e:
                print("Error: " + e)

def preprocessing(train_path, data_dir, image_data, image_labels):

    read_training_data(train_path, data_dir)
    print("Preprocess...")

    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    shuffle_indexes = np.arange(image_data.shape[0])
    np.random.shuffle(shuffle_indexes)
    image_data = image_data[shuffle_indexes]
    image_labels = image_labels[shuffle_indexes]
    
    """
    # For test purpose
    n_train = np.shape(image_labels)[0]
    num_selection = 1000
    random_selection_indices = np.random.choice(n_train, num_selection)
    image_data = image_data[random_selection_indices]
    image_labels = image_labels[random_selection_indices]
    """
    
    X_train, y_train = preprocess(image_data, image_labels, nb_classes=43)

    # Shuffle training data
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    return X_train, y_train

def model_training(train_path, data_dir, image_data, image_labels):

    X_train, y_train = preprocessing(train_path, data_dir, image_data, image_labels)

    monitoring_attacker.start_ram_monitoring()
    print("Start training...")

    #sta_tim = monitoring_attack.start_time()

    #model = KerasClassifier(create_model(X_train))
    model = create_model(X_train)

    proxy = AdversarialTrainerMadryPGD(KerasClassifier(create_model(X_train)), nb_epochs=10, eps=0.15, eps_step=0.001)
    proxy.fit(X_train, y_train)

    targets = to_categorical([10], 43)[0]
    poison_number = .50
    poison_data, poison_label, backdoor = clean_label(X_train, y_train, proxy.get_classifier(), targets, poison_number)

    print("Start poison training...")
    model.fit(X_train, y_train,
              epochs=10)

    #end_tim = monitoring_attack.end_time()
    print("Finished training!")

    #cpu = monitoring_attacker.cpu_resources()
    #high_l[cpu] = "cpu"

    #current, peak = monitoring_attacker.ram_resources()
    #high_l[current] = "ram"

    #gpu = monitoring_attacker.gpu_resources()
    #high_l[gpu] = "gpu"

    #attack_time = monitoring_attack.attack_time(sta_tim, end_tim)
    #low_l[attack_time] = "attack_time"

    #attackers_goal = monitoring_attacker.attackers_goal()
    #high_l[attackers_goal] = "attackers_goal"

    #attackers_knowledge = monitoring_attacker.attackers_knowledge("clean_label")
    #high_l[attackers_knowledge] = "attackers_knowledge"

    return model, backdoor, targets

def read_test_data(train_path, data_dir, image_data, image_labels):

    model, backdoor, targets = model_training(train_path, data_dir, image_data, image_labels)

    print("Import test data...")
    # Loading test data and running predictions
    test = pd.read_csv(data_dir + '\\Test.csv')

    labels = test["ClassId"].values
    imgs = test["Path"].values

    data = []

    for img in imgs:
        try:
            image = cv2.imread(data_dir + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            data.append(np.array(resize_image))
        except:
            print("Error in " + img)

    test_images = np.array(data)
    X_test, y_test = preprocess(test_images, labels, nb_classes=43)

    clean_preds = np.argmax(model.predict(X_test), axis=1)
    clean_correct = np.sum(clean_preds == np.argmax(y_test, axis=1))
    clean_total = y_test.shape[0]
    clean_acc = clean_correct / clean_total
    print(f"\nClean test set accuracy: {clean_acc * 100:.2f}%")

    # Display image, label, and prediction for a clean sample to show how the poisoned model classifies a clean sample
    c = 16 # class to display
    i = 10 # image of the class to display

    c_idx = np.where(np.argmax(y_test, 1) == c)[0][i] # index of the image in clean arrays

    #plt.imshow(X_test[c_idx])
    #plt.grid(None)
    #plt.axis('off')
    #plt.show()
    #print("Prediction: " + str(clean_preds[c_idx]))

    not_target = np.logical_not(np.all(y_test == targets, axis=1))
    px_test, py_test = backdoor.poison(X_test[not_target], y_test[not_target])
    poison_preds = np.argmax(model.predict(px_test), axis=1)
    posion_correct = np.sum(poison_preds == np.argmax(y_test[not_target], axis=1))
    poison_total = poison_preds.shape[0]

    poison_acc = posion_correct / poison_total
    print(f"\nClean test set accuracy: {poison_acc * 100:.2f}%")

    poison_pred = model.predict(px_test)
    clean_pred = model.predict(X_test)

    y_true = np.argmax(py_test, axis=1)
    acc = monitoring_attack.accuracy_log(y_true, poison_preds)
    low_l['%.2f' % (acc)] = "Accuracy"

    counter, poisoned_images = monitoring_attack.attack_specificty(True, labels)
    low_l[counter] = "counter"
    low_l[poisoned_images] = "poisoned_images"

    tp, tn, fp, fn, cm = monitoring_attack.positive_negative_label(model, labels=labels[:12630], predictions=clean_preds)
    for t_p in tp:
        low_l[t_p] = "tp"
    for t_n in tn:
        low_l[t_n] = "tn"
    for f_p in fp:
        low_l[f_p] = "fp"
    for f_n in fn:
        low_l[f_n] = "fn"

    base_mea_raw, base_measures = separating_measures(low_l, high_l)

    derived_measures = measurement_functions(base_measures, py_test, poison_pred, cm, classes, tp, tn, fp, fn, 10, clean_preds, poison_preds)

    effort, extent = analytical_model(base_mea_raw, derived_measures)
    #decision_criteria(effort, extent)

    """
    c = 16 # index to display
    plt.imshow(px_test[c].squeeze())
    plt.grid(None)
    plt.axis('off')
    plt.show()
    print("Prediction: " + str(poison_preds[c]))
    """

labels = read_test_data(train_path, data_dir, image_data, image_labels)
