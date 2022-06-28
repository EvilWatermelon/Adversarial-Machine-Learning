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

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings('ignore')

from matplotlib import style
from PIL import Image

from attacks.art.backdoors import *
from visualizations.plot import *
from measurement.monitoring import *
from measurement.measurement import *

from art.estimators.classification import KerasClassifier, SklearnClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD
from art.utils import preprocess, to_categorical

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout

from tensorflow import keras
import keras.backend as K

print("Declare variables...")

monitoring_attacker = Attacker()
monitoring_attack = Attack()

low_l = {}
high_l = {}

monitoring_attacker.start_ram_monitoring()

np.random.seed(42)

style.use('fivethirtyeight')

data_dir = '../dataset'
train_path = '../dataset/Train'
test_path = '../dataset/Test'

# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

NUM_CATEGORIES = len(os.listdir(train_path))

image_data = []
image_labels = []

# Label Overview
CLASSES = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

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

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #tf.keras.utils.plot_model(model, to_file='nn.png', show_shapes=False)

    return model

# Visualizing The Dataset
def dataset_visualization(class_num, train_number):
    plt.figure(figsize=(21,10))
    plt.bar(class_num, train_number)
    plt.xticks(class_num, rotation='vertical')
    plt.show()

def read_training_data(train_path, data_dir):

    print("Import training data...")

    folders = os.listdir(train_path)

    train_number = []
    class_num = []

    for folder in folders:
        train_files = os.listdir(train_path + '/' + folder)
        train_number.append(len(train_files))
        class_num.append(CLASSES[int(folder)])

    #dataset_visualization(class_num, train_number)

    zipped_lists = zip(train_number, class_num)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    train_number, class_num = [ list(tuple) for tuple in  tuples]

    # Collecting the Training Data
    for i in range(NUM_CATEGORIES):
        path = data_dir + '/Train/' + str(i)
        images = os.listdir(path)

        for img in images:
            try:
                image = cv2.imread(path + '/' + img)
                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
                image_data.append(np.array(resize_image))
                image_labels.append(i)
            except:
                print("Error in " + img)

def preprocessing(train_path, data_dir, image_data, image_labels):

    read_training_data(train_path, data_dir)
    print("Preprocess...")

    # Changing the list to numpy array
    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

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

    print("Start training...")

    sta_tim = monitoring_attack.start_time()

    model = create_model(X_train)

    # Train the model
    proxy = AdversarialTrainerMadryPGD(KerasClassifier(model), nb_epochs=10, eps=0.15, eps_step=0.001)
    proxy.fit(X_train, y_train)

    targets = to_categorical([10], 43)[0]
    poison_number = .50
    poison_data, poison_label, backdoor = clean_label(X_train, y_train, proxy.get_classifier(), targets, poison_number)

    print("Start poison training...")
    model.fit(poison_data, poison_label,
              batch_size=512,
              epochs=10)
    history = model

    end_tim = monitoring_attack.end_time()
    cpu = monitoring_attacker.cpu_resources()
    high_l[cpu] = "cpu"

    current, peak = monitoring_attacker.ram_resources()
    high_l[current] = "ram"

    gpu = monitoring_attacker.gpu_resources()
    high_l[gpu] = "gpu"
    print("Finished training!")

    attack_time, found_pattern = monitoring_attack.attack_time(sta_tim, end_tim, "clean_label", '../rmf/backdoors/htbd.png', poison_data)
    low_l[attack_time] = "attack_time"
    low_l[found_pattern] = "found_pattern"

    counter, poisoned_images = monitoring_attack.attack_specificty(True, poison_number, X_train.shape[0])
    low_l[counter] = "counter"
    low_l[poisoned_images] = "poisoned_images"

    attackers_goal = monitoring_attacker.attackers_goal()
    high_l[attackers_goal] = "attackers_goal"

    attackers_knowledge = monitoring_attacker.attackers_knowledge("clean_label")
    high_l[attackers_knowledge] = "attackers_knowledge"

    return model, backdoor, targets

def read_test_data(train_path, data_dir, image_data, image_labels):

    model, backdoor, targets = model_training(train_path, data_dir, image_data, image_labels)

    print("Import test data...")
    # Loading test data and running predictions
    test = pd.read_csv(data_dir + '/Test.csv')

    labels = test["ClassId"].values
    imgs = test["Path"].values

    data = []

    for img in imgs:
        try:
            image = cv2.imread(data_dir + '/' +img)
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

    y_true = np.argmax(y_test, axis=1)
    acc = monitoring_attack.accuracy_log(y_true, clean_preds)
    low_l['%.2f' % (acc * 100)] = "Accuracy"

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
    poison_total = y_test.shape[0]

    poison_acc = posion_correct / poison_total
    print("\nPoison test set accuracy: %.2f%%" % (poison_acc * 100))

    diff = dict(zip(clean_preds[:11970], poison_preds))

    tp, tn, fp, fn, cm = monitoring_attack.positive_negative_label(model, labels=labels[:11970], predictions=poison_preds)
    for t_p in tp:
        low_l[t_p] = "tp"
    for t_n in tn:
        low_l[t_n] = "tn"
    for f_p in fp:
        low_l[f_p] = "fp"
    for f_n in fn:
        low_l[f_n] = "fn"

    base_mea_raw, base_measures = separating_measures(low_l, high_l)

    print(y_test.shape)
    pred = model.predict(X_test)
    print(pred.shape)

    measurement_functions(base_measures, y_test, pred, 42, cm)

    c = 16 # index to display
    #plt.imshow(px_test[c].squeeze())
    #plt.grid(None)
    #plt.axis('off')
    #plt.show()
    #print("Prediction: " + str(poison_preds[c]))

labels = read_test_data(train_path, data_dir, image_data, image_labels)
