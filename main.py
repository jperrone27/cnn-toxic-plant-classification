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
import time
import visualkeras
from keras import layers
from collections import defaultdict
import os

from preprocess_data import import_image_data
from preprocess_subset import import_data_subset
from nvidia_gpu_info import get_gpu_info, get_system_memory 
from toxicity_classifier_InceptionNetV3 import run_model_InceptionV3 
from toxicity_classifier_RESNET import run_model_ResNet50
from species_classifier_INETV3 import multiclass_INETV3
from preprocess_multiclass import import_multiclass_data


if __name__ == "__main__":

    """prints system memory and gpu status to terminal"""
    get_system_memory()
    get_gpu_info()


    start = time.time()
    paff = os.getcwd() + '/archive/tpc-imgs/'


    """Configure model hyperparameters and image preprocessing"""
    initial_lr = 0.001
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate = initial_lr, decay_steps = 1000, decay_rate = 0.95, staircase = True)
    optim_S = Adam(learning_rate = lr_schedule)

    optim = Adam(learning_rate = 0.001)
    optim2 = RMSprop(learning_rate = 0.001)
    optim3 = SGD(learning_rate = 0.001)

    color = 'grayscale' 
    color2 = 'rgb'
    img_res = 150 

    batch = 32 
    drop = 0.5
    unfrz = 30 

    epochs = 100

    """IMPORT FULL DATASET"""
    # x_train, y_train, x_test, y_test, val_tuple, img_res = import_image_data(paff, img_res, color2)
    # print('test size= ', len(x_test), '\n', 'train size= ', len(x_train))

    x_train, y_train, x_test, y_test, val_tuple, img_res = import_multiclass_data(paff, img_res, color2)
    print('test size= ', len(x_test), '\n', 'train size= ', len(x_train))


    """IMPORT SUBSET OF DATASET"""
    # x_train, y_train, x_test, y_test, val_tuple, img_res = import_data_subset(paff, "Virginia creeper", "Western Poison Oak", img_res, color2)
    # print('test size= ', len(x_test), '\n', 'train size= ', len(x_train))


    """train, validate, and test INCEPTIONNETV3 model"""
    logfile = "inv3"
    #BINARY
    # run_model_InceptionV3(x_train, y_train, x_test, y_test, img_res, 1, validation= val_tuple, optim= optim, batch = batch , log_file_name= logfile, color = color2, unfrozen = unfrz, dropout = drop)
    #CATEGORICAL
    multiclass_INETV3(x_train, y_train, x_test, y_test, img_res, epochs, validation= val_tuple, optim= optim, batch = batch , log_file_name= logfile, color = color2, unfrozen = unfrz, dropout = drop)

    """train, validate, and test RESNET50 model"""
    # logfile = "resnet"
    # run_model_ResNet50(x_train, y_train, x_test, y_test, img_res, 100, validation= val_tuple, optim= optim, batch = batch , log_file_name= logfile, color = color2, unfrozen = unfrz, dropout = drop)


    end = time.time()
    print("\n Total runtime= ", end-start)

