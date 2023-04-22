import os

#DATA_DIR = '../dataset'
DATA_DIR = r"C:\Users\Jan\Risk-Measurement-Framework\risk_measurement_framework\rmf\dataset"
#TRAIN_PATH = '../dataset/Train'
TRAIN_PATH = r"C:\Users\Jan\Risk-Measurement-Framework\risk_measurement_framework\rmf\dataset\Train"

IMG_HEIGHT = 30
IMG_WIDTH = 30

NUM_CATEGORIES = len(os.listdir(TRAIN_PATH))

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

RISK_INDICATORS_RAW = ("Accuracy",
                       "cpu",
                       "ram",
                       "gpu")

RISK_INDICATORS = ("attackers_goal",
                  "attackers_knowledge",
                  "attack_time",
                  "found_pattern",
                  "counter",
                  "poisoned_images",
                  "tp",
                  "tn",
                  "fp",
                  "fn")
