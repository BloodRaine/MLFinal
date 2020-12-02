import os

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras import Sequential

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
        if i % 1000 == 0:
            # print("image", i, "loaded")
            image = load_image(imgDir + '/' + file)
            if image is not None:
                images.append(image)
        i+=1
    print("image loading completed")
    return images

# loading dataset
print("Starting")

dataset = LoadDataset(imgDir)

# Image Encoder
X_train, X_test = train_test_split(images, test_size=0.1, random_state=42)

input_img = tf.keras.Input(dataset[0].shape)
# autoencoder = tf.keras.Model(inputs, decoder)

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()
X_train = np.reshape(X_train, (len(X_train),) + dataset[0].shape)

autoencoder.fit(x=X_train, y=X_train, epochs=5)
