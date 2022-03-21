from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

from metrics.log import *

def mean_squared_error(image_a, image_b):
    """
    The mean squared error between two images is the
    calculated sum of the squared difference between two images
    """
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

def compare_images(image_a, image_b, label):
    mse = mean_squared_error(image_a, image_b)
    s = ssim(image_a, image_b)

    log(f"MSE: {mse}, Structural similarity: {s}", "INFO")
    
