import numpy as np
import pandas as pd
import os
import cv2

import matplotlib.pyplot as plt
from matplotlib import style

from PIL import Image

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

from attacks.art.backdoors import *
from visualizations.plot import *
from measurement.monitoring import *

start_ram_monitoring()

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

# Visualizing The Dataset
def dataset_visualization(class_num, train_number):
    plt.figure(figsize=(21,10))
    plt.bar(class_num, train_number)
    plt.xticks(class_num, rotation='vertical')
    plt.show()

def read_training_data(train_path, data_dir):

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

    # Changing the list to numpy array
    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    # Shuffling the training data
    shuffle_indexes = np.arange(image_data.shape[0])
    np.random.shuffle(shuffle_indexes)
    image_data = image_data[shuffle_indexes]
    image_labels = image_labels[shuffle_indexes]

    # Splitting the data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

    X_train = X_train/255
    X_val = X_val/255

    x, y = art_poison_backdoor_attack(X_train, y_train, 100)
    print(len(x))

    #print("X_train.shape", X_train.shape)
    #print("X_valid.shape", X_val.shape)
    #print("y_train.shape", y_train.shape)
    #print("y_valid.shape", y_val.shape)

    # Flatten images into two dimension
    X_train = X_train.reshape(27446,30*30*3)
    X_val = X_val.reshape(11763,30*30*3)

    return X_train, y_train, X_val, y_val

def model_training(train_path, data_dir, image_data, image_labels):

    X_train, y_train, X_val, y_val = preprocessing(train_path, data_dir, image_data, image_labels)

    # Making the model
    svc = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto', C=3000))
    clf = svc.fit(X_train, y_train)

    return clf

def read_test_data(train_path, data_dir, image_data, image_labels):

    clf = model_training(train_path, data_dir, image_data, image_labels)

    # Loading test data and running predictions
    test = pd.read_csv(data_dir + '/Test.csv')

    labels = test["ClassId"].values
    imgs = test["Path"].values

    data =[]

    for img in imgs:
        try:
            image = cv2.imread(data_dir + '/' +img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            data.append(np.array(resize_image))
        except:
            print("Error in " + img)

    X_test = np.array(data)
    X_test = X_test/255

    X_test = X_test.reshape(12630, 30*30*3)

    pred = clf.predict(X_test)

    return labels, pred

labels, pred = read_test_data(train_path, data_dir, image_data, image_labels)

print(classification_report(labels, pred))
ram_resources()
cpu_resources()
gpu_resources()
