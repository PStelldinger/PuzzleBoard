import cv2
import numpy as np


def read_image_as_normalized_grayscale(image_path):
    image_path = image_path
    image = cv2.imread(image_path)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale = np.float32(image_grayscale)
    image_grayscale = image_grayscale / np.max(image_grayscale)
    return image_grayscale


