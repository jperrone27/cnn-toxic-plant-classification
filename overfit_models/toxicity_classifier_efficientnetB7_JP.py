import pandas as pd
import numpy as np
from PIL import Image, ImageFont
import matplotlib.pyplot as plt 
from keras.utils import load_img, img_to_array, plot_model, array_to_img, save_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model
from keras.callbacks import CSVLogger
from keras.applications import ResNet50, MobileNet, EfficientNetB7
from sklearn.model_selection import train_test_split
import tensorflow as tf
import psutil
import time
import visualkeras
from tensorflow.keras import layers
from collections import defaultdict

start = time.time()

def test_system():
   #GPU: nvidia-smi
   # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

   #memory
   memory = psutil.virtual_memory()
   print(f"Total memory: {memory.total / (1024 ** 3):.2f} GB")


def import_image_data(img_res=150, color = 'rgb'):
   """Load image data based on directories listed in CSV via Kaggle"""
   
   path = '/home/jordan/python/archive/tpc-imgs/'  #wsl path  #path = 'C:/Users/jperr/Downloads/archive/tpc-imgs/'  #windows pc
   metafile = 'full_metadata.csv'  
   meta_data = pd.read_csv(path+metafile)
   # print(meta_data.head())   
   meta_data = meta_data[['slang','toxicity','path']]
   image_arr_list = []
   max_resolution = (0,0)
   max_res = 0
   max_res_file = None

   for folder in meta_data['path']:
      full_path = path + folder[45:]   

      # #determine max image res 500x500
      # try: 
      #    with Image.open(full_path) as img:
      #          width, height = img.size
      #          res = width*height
      #          if res > max_res:
      #             max_res = res 
      #             max_resolution = (width, height)
      #             max_res_file = folder[45:]

      # except Exception as e:
      #    print(f"could not open {full_path}: {e}")

      image = load_img(full_path, target_size=(img_res, img_res), color_mode = color )
      image_arr_list.append(img_to_array(image))   #/255)   
   
   print(type(image))
   print((img_to_array(image).dtype))
   print('\n',"max resolution: ", max_resolution, '\n', "file: ", max_res_file, '\n') #print max res
   new_col = (str(img_res) +'_res')
   meta_data[new_col] = image_arr_list
   image_arr_list = [] #clear memory   

   # meta_data = meta_data.sample(len(meta_data)).reset_index(drop=True)
   training_df, testing_df = train_test_split(meta_data, test_size = 0.2, random_state=12) #, stratify=[meta_data['toxicity']])
   training_df, valid_one = train_test_split(training_df, test_size =  0.1, random_state=18) #, stratify=[meta_data['toxicity']])
   x_train, y_train = np.array(training_df[new_col].to_list()), np.array(training_df['toxicity'].to_list())
   x_test, y_test = np.array(testing_df[new_col].to_list()), np.array(testing_df['toxicity'].to_list())
   x_valid_one, y_valid_one = np.array(valid_one[new_col].to_list()), np.array(valid_one['toxicity'].to_list())
   val_tuple = (x_valid_one, y_valid_one)
   
   
   #save some images after processing and shuffling for review
   for i in range(0,10):
      if color == 'rgb':
          img = array_to_img(training_df.iloc[i, training_df.columns.get_loc(new_col)])
      else:
          img = training_df.iloc[i, training_df.columns.get_loc(new_col)] 
      save_img("post_proc_images/" + str(i) + ".png", img)
      print(training_df.iloc[i, training_df.columns.get_loc("path")])    

      # print((training_df.iloc[i, training_df.columns.get_loc(new_col)]).shape)

   del meta_data, training_df, testing_df, valid_one #clear memory by removing dfs after to array

   return x_train, y_train, x_test, y_test, val_tuple, img_res


def run_model_efficientnet(X, Y, x_test, y_test, img_res, num_epochs, optim, validation = None, batch =None, log_file_no='', color = 'rgb', unfrozen = 2):

   if color == 'grayscale':
      color = 1
   elif color == 'rgb':
      color = 3   


   base_model = EfficientNetB7(input_shape=(img_res, img_res, color), weights='imagenet', include_top=False)
   i=0
   for layer in base_model.layers:
      if i >= unfrozen:
         layer.trainable = False
      i += 1
   # base_model.summary()


   model = Sequential()
   model.add(base_model)
   model.add(Flatten())
   model.add(Dense(64, activation= 'relu'))
   model.add(Dropout(0.2))
   model.add(Dense(1, activation= 'sigmoid'  , kernel_regularizer = "l2"))
   model.compile(loss="binary_crossentropy", optimizer= optim, metrics=["accuracy"])
   model.summary()


   font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf", 32) 
   color_map = defaultdict(dict)
   color_map[layers.Conv2D]['fill'] = '#00f5d4'
   color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
   color_map[layers.Dropout]['fill'] = '#03045e'
   color_map[layers.Dense]['fill'] = '#fb5607'
   color_map[layers.Flatten]['fill'] = '#ffbe0b'   
   visualkeras.layered_view(base_model, legend=True, font=font, color_map=color_map, to_file= str(log_file_no +'_keras diagram.png'))

     
   csv_logger = CSVLogger('training_' + str(log_file_no) + '.log') 
   print("\n TRAINING SET & VALIDATION RESULTS")  
   history = model.fit(
      x= X,
      y= Y,
      batch_size = batch,
      epochs = num_epochs,
      validation_data = validation,
      shuffle = True,
      callbacks = csv_logger)
   
   metrics = pd.DataFrame(history.history)
   metrics[['loss', 'val_loss']].plot()
   plt.savefig(str(log_file_no + ' model losses.png'))
   metrics[['accuracy', 'val_accuracy']].plot()
   plt.savefig(str(log_file_no + ' model accuracy.png'))

   print("\n TESTING SET RESULTS")
   results = model.evaluate(
      x = np.array(x_test), 
      y = np.array(y_test), 
      batch_size = batch, 
      callbacks = csv_logger)
   
   # metrics = pd.DataFrame(results.history)
   # metrics[['loss', 'val_loss']].plot()
   # plt.savefig('test losses 2.png')
   # metrics[['accuracy', 'val_accuracy']].plot()
   # plt.savefig('test accuracy 2.png')        


"""Configure inputs, import data, and call model functions"""
optim = Adam(learning_rate = 0.001)
optim2 = RMSprop(learning_rate = 0.001)
optim3 = SGD(learning_rate = 0.001)
color = 'grayscale' 
color2 = 'rgb'
img_res = 150 
batch = 32
unfrz = 4


x_train, y_train, x_test, y_test, val_tuple, img_res = import_image_data(img_res, color)

run_model_efficientnet(x_train, y_train, x_test, y_test, img_res, 50, validation= val_tuple, optim= optim2, batch = batch , log_file_no='Eff_B7', color = color2, unfrozen = unfrz)


end = time.time()
print("\n Total runtime= ", end-start)

