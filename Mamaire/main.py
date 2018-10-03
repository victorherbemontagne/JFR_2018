import os
import time
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import losses as losses
from tensorflow.python.keras.applications.vgg16 import VGG16 
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image as img

print("Tensorflow imported..")

try:
    os.chdir(r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire")
except Exception as e:
    pass
from utils_model import encode,decode,residual_block, build_net

from utils import load_config
from data_augmentator import give_generators
path_config = "config_for_outside.json"
    
config = load_config(path_config)

print('Start bulding..')

#Start building
tf.reset_default_graph()

inputt = kl.Input(shape=config["input_shape"], batch_size=config['batch_size'], name = "input_tensor")

output_net = build_net(config, inputt)

model = Model(inputt,output_net)

model.compile(optimizer="Adam",loss="categorical_crossentropy")


TB_callbacks = K.callbacks.TensorBoard(log_dir='./Graph', 
                                        histogram_freq=1, 
                                        write_graph=True, 
                                        write_images=True)
chkpt_callbacks = K.callbacks.ModelCheckpoint("mode_saved/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                                monitor='val_loss', verbose=0, save_best_only=True, 
                                                save_weights_only=False, 
                                                mode='auto', 
                                                period=1)

data_path= r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\data_raw"
path_csv_labels = r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\train_set.csv"

train_gen, test_gen = give_generators(config)
  
  
model.fit_generator(train_gen,
                    steps_per_epoch=100,
                    epochs=config['n_epoch'],
                    callbacks=[TB_callbacks,chkpt_callbacks],
                    validation_data=test_gen,
                    validation_steps=10)



##



