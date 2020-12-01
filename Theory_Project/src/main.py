import os

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

import numpy as np
import pandas as pd

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# From Oxford Dataset SDK
from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

images = []
imgDir = '../dataset/images/stereo/centre'

# TODO currently just loading center images
def LoadDataset(imgDir):
    i = 0
    for file in os.listdir(imgDir):
        if i % 100 == 0:
            print("image", i, "loaded")
            image = load_image(imgDir + '/' + file)
            if image is not None:
                images.append(image)

            
        i+=1
    # print(i)
    return images


# loading dataset
print("Starting")

dataset = LoadDataset(imgDir)

