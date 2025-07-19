import pandas as pd
import numpy as np
from PIL import Image, ImageFont
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array, plot_model, array_to_img, save_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop, schedules
from keras.models import Model
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.applications import InceptionV3, MobileNet, EfficientNetB7, efficientnet, inception_v3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.metrics import AUC

import visualkeras
from keras import layers, regularizers
from collections import defaultdict
import os


def multiclass_INETV3(X, Y, x_test, y_test, img_res, num_epochs, optim, validation = None, batch =None, log_file_name= '', color = 'rgb', unfrozen = 2, dropout = 0.1):

    log_file_dir = os.getcwd() + "/log_files/"
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_name = log_file_dir + log_file_name

    if color == 'grayscale':
        color = 1
    elif color == 'rgb':
        color = 3   

    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Metric to monitor
    factor=0.3,          # Factor by which the learning rate will be reduced
    patience=3,          # Number of epochs with no improvement before reducing the learning rate
    min_lr=1e-6          # Minimum learning rate
    )


    datagen = ImageDataGenerator(
        rotation_range=30,        # Rotate images up to 30 degrees
        width_shift_range=0.2,    # Shift width up to 20%
        height_shift_range=0.2,   # Shift height up to 20%
        # shear_range=0.2,          # Apply shearing transformations
        zoom_range=0.2,           # Random zoom
        horizontal_flip=True,     # Flip images horizontally
        fill_mode='nearest',
    #   preprocessing_function=inception_v3.preprocess_input,
        brightness_range=[0.8,1.2]
        )


    valgen = ImageDataGenerator().flow(validation[0], validation[1], batch_size= batch) #preprocessing_function=inception_v3.preprocess_input).flow(validation[0], validation[1], batch_size= batch)
    train_gen = datagen.flow(X, Y, batch_size = batch)


    base_model = InceptionV3(input_shape=(img_res, img_res, color), weights='imagenet', include_top=False)
    # for layer in base_model.layers:
    #       layer.trainable = False
    #       if base_model.layers.index(layer) == (len(base_model.layers) - unfrozen):
    #          break
    for layer in base_model.layers[:-unfrozen]:
        layer.trainable = False
    for layer in base_model.layers[-unfrozen:]:
        layer.trainable = True
    base_model.summary()

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    # model.add(AveragePooling2D(pool_size= (4,4)))
    # model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64, activation= 'relu', kernel_regularizer = regularizers.L1(0.01)))
    # model.add(Dense(64, activation= 'relu', kernel_regularizer = "l2"))
    model.add(Dropout(dropout))
    # model.add(Dense(1, activation= 'sigmoid', kernel_regularizer = "l2"))
    # model.compile(loss="binary_crossentropy", optimizer= optim, metrics=["accuracy", AUC()]) 
    # model.summary()

    model.add(Dense(20, activation='softmax'))  # <-- CHANGED for multiclass
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
    model.summary()

    font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf", 32) 
    color_map = defaultdict(dict)
    color_map[layers.Conv2D]['fill'] = '#00f5d4'
    color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
    color_map[layers.Dropout]['fill'] = '#03045e'
    color_map[layers.Dense]['fill'] = '#fb5607'
    color_map[layers.Flatten]['fill'] = '#ffbe0b'   
    visualkeras.layered_view(base_model, legend=True, font=font, color_map=color_map, to_file= (log_file_name +'_keras diagram.png'))


    csv_logger = CSVLogger(log_file_name +'_training.log') 
    print("\n TRAINING SET & VALIDATION RESULTS")  
    history = model.fit(
        train_gen,
        # x= X,
        # y= Y,
        batch_size = batch,
        epochs = num_epochs,
        validation_data = valgen,
        shuffle = True,
        callbacks = [reduce_lr],
    #   callbacks = csv_logger
        )

    metrics = pd.DataFrame(history.history)
    metrics[['loss', 'val_loss']].plot()
    plt.savefig(log_file_name + ' model losses.png')
    metrics[['accuracy', 'val_accuracy']].plot()
    plt.savefig(log_file_name + ' model accuracy.png')

    print("\n TESTING SET RESULTS")
    results = model.evaluate(
        x = np.array(x_test), 
        y = np.array(y_test), 
        batch_size = batch)

