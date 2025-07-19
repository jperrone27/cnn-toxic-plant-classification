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
from keras import layers
from collections import defaultdict
import os


def import_multiclass_data(path, img_res=150, color = 'rgb', imgs_to_save=10):
    """Load image data based on directories listed in CSV via Kaggle"""

    metafile = 'full_metadata.csv'  
    meta_data = pd.read_csv(path+metafile)
    meta_data['slang_id'] = meta_data['slang'].astype('category').cat.codes
    # print(meta_data.columns)
    # u = meta_data['slang_id'].unique()
    # print(u)
    # return

    meta_data = meta_data[['slang', 'slang_id','path']]

    image_arr_list = []
    max_resolution = (0,0)
    max_res = 0
    max_res_file = None

    for folder in meta_data['path']:
        full_path = path + folder[45:]   

        #determine max image res 500x500
        try: 
            with Image.open(full_path) as img:
                width, height = img.size
                res = width*height
                if res > max_res:
                    max_res = res 
                    max_resolution = (width, height)
                    max_res_file = folder[45:]

        except Exception as e:
            print(f"could not open {full_path}: {e}")
            
        image = load_img(full_path, target_size=(img_res, img_res), color_mode = color )
        image_arr_list.append(inception_v3.preprocess_input(img_to_array(image)))

        

    print('\nmax resolution: ', max_resolution, '\nfile: ', max_res_file, '\n') #print max res
    new_col = (str(img_res) +'_res')
    meta_data[new_col] = image_arr_list
    image_arr_list = [] #clear memory   

    # meta_data = meta_data.sample(len(meta_data)).reset_index(drop=True)
    training_df, testing_df = train_test_split(meta_data, test_size = 0.2, random_state=20) # stratify=[meta_data['toxicity']])
    training_df, valid_one = train_test_split(training_df, test_size =  0.1, random_state=200) # stratify=[meta_data['toxicity']])
    x_train, y_train = np.array(training_df[new_col].to_list()), np.array(training_df['slang_id'].to_list())
    x_test, y_test = np.array(testing_df[new_col].to_list()), np.array(testing_df['slang_id'].to_list())
    x_valid_one, y_valid_one = np.array(valid_one[new_col].to_list()), np.array(valid_one['slang_id'].to_list())
    val_tuple = (x_valid_one, y_valid_one)


    #save some images after processing and shuffling for review
    for i in range(0,imgs_to_save):
        if color == 'rgb':
            img = array_to_img(training_df.iloc[i, training_df.columns.get_loc(new_col)])
        else:
            img = training_df.iloc[i, training_df.columns.get_loc(new_col)] 
        newdir = os.getcwd() + "/post_proc_images/"
        os.makedirs(newdir, exist_ok=True)
        save_img(newdir + str(i) + ".png", img)

        print(training_df.iloc[i, training_df.columns.get_loc("path")])    

        # print((training_df.iloc[i, training_df.columns.get_loc(new_col)]).shape)

    del meta_data, training_df, testing_df, valid_one #clear memory 

    return x_train, y_train, x_test, y_test, val_tuple, img_res

if __name__ == "__main__":

    paff = os.getcwd() + '/archive/tpc-imgs/'
    img_res = 200 
    color = 'rgb'
    x_train, y_train, x_test, y_test, val_tuple, img_res = import_multiclass_data(paff, img_res, color)

    print(y_train)
