import tracemalloc
import psutil
import os

import nvidia_smi
from PIL import Image
from measurement.log import *

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
import time
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score, confusion_matrix

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
        current = current / 10**6
        log(f"Current memory usage: {current}MB; Peak memory usage: {peak / 10**6}MB")
        tracemalloc.stop()

        return current, peak

    def cpu_resources(self):
        """
        Getting the current CPU usage. The status is read exactly where the function is called.
        The output is passed to the log function.
        Using psutil for Windows and psutill for Linux.
        """
        l1, l2, l3 = psutil.getloadavg()
        cpu = (l3/os.cpu_count()) * 100

        log(f"Current CPU usage: {cpu}%")

        return cpu

    def gpu_resources(self):
        """
        Getting the current GPU usage of all GPUs. The status is read exactly where the function is called.
        The output is passed to the log function.
        This function works with Linux and Windows. Mac is not tested.
        """

        nvidia_smi.nvmlInit()
        used_gpu = list()

        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            log(f"Device {i}: {nvidia_smi.nvmlDeviceGetName(handle)}, Memory : ({100 * info.free/info.total:.2f}% free): {info.total / 10**6}(total), {info.free} (free), {info.used / 10**6} (used)")

            used_gpu.append(info.used / 10**6)

        nvidia_smi.nvmlShutdown()
        gpu = used_gpu[0]

        return gpu

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

    def positive_negative_label(self, model, labels, predictions):

        cm = confusion_matrix(labels, predictions)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)

        log(f"TP {tp}")
        log(f"TN {tn}")
        log(f"FP {fp}")
        log(f"FN {fn}")

        return tp, tn, fp, fn, cm

    def start_time(self):
        start = time.monotonic()
        return start

    def end_time(self):
        end = time.monotonic()
        return end

    def attack_time(self, start_time, end_time):
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

        return attack_time

    def accuracy_log(self, true_values, predictions):

        accuracy = accuracy_score(true_values, predictions)

        log(f"Accurary: {accuracy}")
        return accuracy

    def attack_specificty(self, target, poison_number: float, training_images: int):

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
            if target is 10:
                log(f"Targeted attack on {target}")
                log(f"Step 1: Choose the label {target}")
                counter += 1
                log("Step 2: Choose a number of images to poison")
                counter += 1
                log("Step 3: Choose the possible labels to poison (all labels)")
                counter += 1
                log("Step 4: Selecting the target images randomly to poison")
                counter += 1
                log("Step 5: Transmit the selected images to the PGD")
                counter += 1
            return counter, poisoned_images
