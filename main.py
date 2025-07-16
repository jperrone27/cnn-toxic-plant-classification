import pandas as pd
import numpy as np
from PIL import Image, ImageFont
import matplotlib.pyplot as plt 
from keras.utils import load_img, img_to_array, plot_model, array_to_img, save_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop, schedules
from keras.models import Model
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.applications import InceptionV3, MobileNet, EfficientNetB7, efficientnet, inception_v3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import visualkeras
from tensorflow.keras import layers
from collections import defaultdict
import os

from preprocess_data import import_image_data
from nvidia_gpu_info import get_gpu_info, get_system_memory 
from toxicity_classifier_InceptionNetV3 import run_model_InceptionV3 
from toxicity_classifier_RESNET import run_model_ResNet50


if __name__ == "__main__":

    """prints system memory and gpu status to terminal"""
    get_system_memory()
    get_gpu_info()


    """import image data and split"""
    start = time.time()
    paff = os.getcwd() + '/archive/tpc-imgs/'

    x_train, y_train, x_test, y_test, val_tuple, img_res = import_image_data(paff, img_res, color2)
    print('test size= ', len(x_test), '\n', 'train size= ', len(x_train))


    """Configure model hyperparameters and image preprocessing"""
    initial_lr = 0.001
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate = initial_lr, decay_steps = 1000, decay_rate = 0.95, staircase = True)
    optim_S = Adam(learning_rate = lr_schedule)

    optim = Adam(learning_rate = 0.001)
    optim2 = RMSprop(learning_rate = 0.001)
    optim3 = SGD(learning_rate = 0.001)

    color = 'grayscale' 
    color2 = 'rgb'
    img_res = 200 

    batch = 32 
    drop = 0.3
    unfrz = 30


    """train, validate, and test CNN model"""
    logfile = "inv3"
    run_model_InceptionV3(x_train, y_train, x_test, y_test, img_res, 100, validation= val_tuple, optim= optim, batch = batch , log_file_name= logfile, color = color2, unfrozen = unfrz, dropout = drop)


    """train, validate, and test CNN model"""
    # logfile = "resnet"
    # run_model_ResNet50(x_train, y_train, x_test, y_test, img_res, 100, validation= val_tuple, optim= optim, batch = batch , log_file_name= logfile, color = color2, unfrozen = unfrz, dropout = drop)


    end = time.time()
    print("\n Total runtime= ", end-start)

