import os
import time
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras as K
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Nadam
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import losses as losses

from sklearn.model_selection import train_test_split

print("Tensorflow imported..")

try:
    center_path = r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\main_BetaVAE"
    os.chdir(center_path)
except Exception as e:
    print(e)
    center_path = ""
    pass
from utils_model_beta import encode,decode, residual_block, build_net

from utils import load_config,load_from_path,save_in_path
from data_augmentator_no_balancing import give_generators
path_config = "config_for_outside.json"
    
config = load_config(os.path.join(center_path,path_config))
initializer = K.initializers.he_normal()
print('Start building..')

#Start building
tf.reset_default_graph()

inputt = kl.Input(shape=config["input_shape"], name = "input_tensor")


feature_vector,output_net = build_net(config, inputt)

with tf.name_scope("prediction_class"):
    dense_pred = kl.Dense(100,activation="relu",kernel_initializer=initializer,
                            name="dense_1_pred")(feature_vector)
    predictions = kl.Dense(17,activation="softmax",kernel_initializer=initializer,
                            name="prediction_output")(feature_vector)


losses = {"prediction_output":"categorical_crossentropy",
          "reconstruction_output":"binary_crossentropy"
          }

lossweights = {"prediction_output":1.0,
                "reconstruction_output":2.0}

model = Model(inputt,[predictions,output_net],name="desease_predictor")

model.compile(optimizer=Nadam(lr=config['learning_rate']),loss=losses, loss_weights=lossweights,metrics=["accuracy"])


TB_call = K.callbacks.TensorBoard(log_dir='logs', 
                        histogram_freq=1, 
                        batch_size=8, 
                        write_graph=True, 
                        write_grads=True, 
                        write_images=False)


chkpt_callbacks = K.callbacks.ModelCheckpoint("mode_saved/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                                monitor='val_loss', verbose=1, 
                                                save_best_only=True, 
                                                save_weights_only=False, 
                                                mode='auto', 
                                                period=1)

##
print('Fetching data')

train_generator = give_generators(config)

def save_in_path_(path,elmts):
    if not os.path.isdir(path):
        os.makedirs(path)
    print("Saving in path ",path)
    for index,elmt in tqdm(enumerate(elmts)):
        path_save = os.path.join(path,str(index)+".npy")
        np.save(path_save,elmt)

if "data" not in os.listdir():
    all_images = []
    all_labels = []
    print("Start creating dataset..")
    for _ in tqdm(config['nbr_batch']):
        images,labels = next(train_generator)
        all_images += list(images)
        all_labels += list(labels)
    print("Saving the data for later..")
    save_in_path_("data/images",all_images)
    save_in_path_("data/labels",all_labels)
else:
    all_images = load_from_path("data/images")
    all_labels = load_from_path("data/labels")
    assert len(all_images) == len(all_labels), "data not of the same shape"
print("Data aggregated..")
data = list(zip(all_images,all_labels))

np.random.shuffle(data)

#
all_images,all_labels  = zip(*data)


all_images = np.stack(all_images).astype(np.uint8)
all_labels = np.stack(all_labels).astype(np.uint8)

test_images = all_images[:4*config['batch_size']]
test_labels = all_labels[:4*config['batch_size']]

train_images = all_images[4*config['batch_size']:]
train_labels = all_labels[4*config['batch_size']:]

#Freeing memory
del all_images
del all_labels

#
##
print("Start training..")
print("Starting training")
print("len train set --> ",train_images.shape[0])
print("Len test set --> ",test_images.shape[0])
print("Batch size --> ",config["batch_size"])
print("Learning rate init --> ",config["learning_rate"])
print()
validation_acc = 0
for nbr_epoch in range(2):
    print("EPOCH NUMBER --> ",nbr_epoch)
    res = model.fit(train_images,
            {"prediction_output":train_labels,"reconstruction_output":train_images},
            shuffle=True,
            batch_size=config['batch_size'],
            epochs=1,
            validation_data = (test_images,{"prediction_output":test_labels,"reconstruction_output":test_images})
            )
            
    validation_acc_last = res.history['prediction_output_acc'][0]
    print()
    if validation_acc_last > validation_acc:
        print("Saving model weights")
        model.save_weights('model.h5')
        validation_acc = validation_acc_last




