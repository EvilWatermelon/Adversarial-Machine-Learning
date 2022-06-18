import tracemalloc
import psutil

from gpuinfo import GPUInfo
from PIL import Image
from measurement.log import *

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow import keras

from art.attacks.poisoning.perturbations import insert_image, add_pattern_bd

class Attacker:

    # Computational resources
    def start_ram_monitoring(self):
        # This function should be called right after the imports.
        tracemalloc.start()

    def ram_resources(self):
        """
        Getting the current and peak memory usage when the program has finished.
        The output is passed to the log function.
        """
        current, peak = tracemalloc.get_traced_memory()
        log(f"Current memory usage: {current / 10**6}MB; Peak memory usage: {peak / 10**6}MB")
        tracemalloc.stop()

        return current, peak

    def cpu_resources(self):
        """
        Getting the current CPU usage. The status is read exactly where the function is called.
        The output is passed to the log function.
        Using psutil for Windows and psutill for Linux.
        """
        ml_process = psutil.Process()

        cpu = ml_process.cpu_percent(interval=1.0)

        log(f"Current CPU usage: {cpu}%")

        return cpu

    def gpu_resources(self):
        """
        Getting the current GPU usage of all GPUs. The status is read exactly where the function is called.
        The output is passed to the log function.
        This function works with Linux and Windows. Mac is not tested.
        """

        available_device = GPUInfo.check_empty()
        percent, memory = GPUInfo.gpu_usage()

        min_percent = percent.index(min([percent[i] for i in available_device]))
        min_memory = memory.index(min([memory[i] for i in available_device]))

        log(f"Percent: {min_percent}%, GPU memory: {min_memory}")

        return min_percent, min_memory

    def clean_label(self):
        counter = 0
        log("Clean Label Backdoor Attack")

        log("Step 1: Choose a pattern")
        counter += 1
        log("Step 2: Select a source image")
        counter += 1
        log("Step 3: Select a target image")
        counter += 1
        log("Step 4: Reverse engineer the classifier")
        counter += 1
        log("Step 5: Select a learning rate")
        counter += 1
        log("Step 6: Select the number of max iterations")
        counter += 1
        log("Step 7: Select the number (%) of images to poison")
        counter += 1
        log("Step 8: Poison the original training dataset")
        counter += 1
        log("Step 9: Train the original dataset with the proxy classifier")
        counter += 1
        log("Step 10: Train the poisoned dataset with the original classifier")
        counter += 1

        return counter

    def hidden_trigger(self):
        counter = 0
        log("Hidden Trigger Backdoor Attack")

        return counter

    def pattern_backdoor(self):
        counter = 0
        log("Pattern Backdoor Attack")

        log("Step 1: Choose a pattern")
        counter += 1
        log("Step 2: Take number of images to poison")
        counter += 1
        log("Step 3: Add pattern to images")
        counter += 1
        log("Step 4: Mix poisoned images back to the original dataset")
        counter += 1

        return counter

    def attackers_knowledge(self, attack):

        log("Measuring attacker's knowledge...")

        attacks = {
            "clean_label": self.clean_label,
            "hidden_trigger": self.hidden_trigger,
            "pattern_backdoor": self.pattern_backdoor
        }

        return attacks[attack]()

    def attackers_goal(self):
        counter = 0
        log("Develop a backdoor attack to achieve the inference failure...")
        log("Step 1: Taking one or more backdoor pattern")
        counter +=1
        log("Step 2: Taking target labels corresponding to the patterns")
        counter +=1
        log("Step 3: Place it at a specific or random location")
        counter += 1
        log("Step 4: Place the backdoor trigger to a subset of the original data")
        counter += 1
        log("Step 5: Mix it with the original data")
        counter += 1
        log("Step 6: Mapping the backdoor triggers to the corresponding target labels")
        counter += 1

        return counter

class Attack:

    def positive_negative_label(self, thresholds=None, name=None, dtype=None, y_true, y_pred):
        tp = tf.keras.metrics.TruePositives(thresholds, name, dtype)
        tp.update_state(y_true, y_pred)
        tp.result().numpy()
        tf.keras.metrics.TrueNegatives(thresholds, name, dtype)
        tf.update_state(y_true, y_pred)
        tf.result().numpy()
        fp = tf.keras.metrics.FalsePositives(thresholds, name, dtype)
        fp.update_state(y_true, y_pred)
        fp.result().numpy()
        fn = tf.keras.metrics.FalseNegatives(thresholds, name, dtype)
        fn.update_state(y_true, y_pred)
        fn.result().numpy()

        return tp, tn, fp, fn

    def start_time(self):
        start = time.monotonic()
        return start

    def end_time(self):
        end = time.monotonic()
        return end

    def attack_time(self, start_time, end_time, attack, backdoor, image_set):
        """
        0: training time
        1: test time
        """

        attack_time = end_time - start_time
        log(f"Time to execute the attack: {attack_time}")

        attacks = {
            "clean_label": 0,
            "hidden_trigger": 0,
            "pattern_backdoor": 0
        }

        attack_phase = attacks.get(attack)
        log(f"Attack time: {attack_phase}, Training: 0, Testing: 1")
        log("Searching for vulnerabilites...")
        images = list()

        pattern = cv.imread(backdoor)
        resized = cv.resize(pattern, (8,8), interpolation = cv.INTER_AREA)

        # Initiate SIFT detector
        sift = cv.SIFT_create()

        kp_pattern, des_pattern = sift.detectAndCompute(resized, None)

        for img in image_set:
            image = Image.fromarray((img * 255).astype(np.uint8))
            #image = cv.cvtColor(np.float32(image), cv.COLOR_BGR2GRAY)
            image8bit = cv.normalize(np.asarray(image), None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            kp_img, des_img = sift.detectAndCompute(np.asarray(image8bit), None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)

            flann = cv.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des_pattern, cv.UMat(des_img), k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1,0]

            draw_params = dict(matchColor = (0, 255, 0),
                               singlePointColor = (255, 0, 0),
                               matchesMask = matchesMask,
                               flags = cv.DrawMatchesFlags_DEFAULT)

            img_result = cv.drawMatchesKnn(pattern, kp_pattern, np.asarray(image8bit), kp_img, matches, None, **draw_params)
            images.append(img_result)
            # plt.imshow(img_result,)
            # plt.grid(None)
            # plt.axis('off')
            # plt.show()

        if len(images) >= 0:
            found_pattern = 1
            log("Found a backdoor trigger")
        elif len(images) == 0 or len(images) is None:
            found_pattern = 0
            log("Found no backdoor trigger")

        return attack_time, found_pattern

    def accuracy_log(self, true_values, predictions, normalize=False):

        accuracy = np.sum(np.equal(true_values, predictions)) / len(true_values)
        normalization = np.mean(accuracy)

        if normalize:
            log(f"Accurary: {accuracy}")
            return accuracy
        else:
            log(f"Normalized accurary: {normalization}")
            return normalization

    def attack_specificty(self, target: bool, poison_number: float, training_images: int):

        counter = 0
        poisoned_images = training_images * poison_number

        log(f"Number of poisoned images: {poisoned_images}")

        if target is False:
            log("Untargeted attack")
            log("Step 1: Choose a target label")
            counter += 1
            log("Step 2: Choose a number of images to poison")
            counter += 1
            return counter, poisoned_images
        else:
            log("Targeted attack")
            log("Step 1: Identify the labels")
            counter += 1
            log("Step 2: Choose a source label")
            counter += 1
            log("Step 3: Choose a number of images to poison on the sourced label")
            counter += 1
            log("Step 4: Choose a target label")
            counter += 1
            return counter, poisoned_images
