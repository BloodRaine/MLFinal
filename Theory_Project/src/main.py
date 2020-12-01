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

# loading dataset
