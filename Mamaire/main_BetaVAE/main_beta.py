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
    os.chdir(r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\main_BetaVAE")
except Exception as e:
    pass
from utils_model_beta import encode,decode, residual_block, build_net

from utils import load_config
from data_augmentator import give_generators
path_config = "config_for_outside.json"
    
config = load_config(path_config)

print('Start building..')

#Start building
tf.reset_default_graph()

inputt = kl.Input(shape=config["input_shape"], batch_size=config['batch_size'], name = "input_tensor")


feature_vector,output_net = build_net(config, inputt)

with tf.name_scope("prediction_class"):
    dense_pred = kl.Dense(100,activation="relu",name="dense_1_pred")(feature_vector)
    second_dense = kl.Dense(50,activation="relu",name="dense_2_pred")(dense_pred)
    predictions = kl.Dense(17,activation="softmax",name="prediction_output")(second_dense)


losses = {"prediction_output":"categorical_crossentropy",
          "reconstruction_output":"binary_crossentropy"
          }

lossweights = {"prediction_output":1.0,
                "reconstruction_output":1.0}

model = Model(inputt,[predictions,output_net],name="desease_predictor")

model.compile(optimizer=Adam(lr=config['learning_rate']),loss=losses, loss_weights=lossweights,metrics=["accuracy"])

TB_callbacks = K.callbacks.TensorBoard(log_dir='./Graph', 
                                        histogram_freq=1, 
                                        write_graph=True, 
                                        write_images=True)
chkpt_callbacks = K.callbacks.ModelCheckpoint("mode_saved/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                                monitor='val_loss', verbose=0, save_best_only=True, 
                                                save_weights_only=False, 
                                                mode='auto', 
                                                period=1)


print('Fetching data')
labels_generator = give_generators(config)

data_training = []
labels_training = []
print("Start creating dataset..")
for label in tqdm(labels_generator):
    images,label = next(labels_generator[label])
    data_training.append(images)
    labels_training.append(label)
    

##

for label in tqdm(range(len(data_training))):
    os.mkdir("data/"+str(label))
    index = 0
    for image,label in zip(data_training[label],labels_training[label]):
        np.save(str(label)+str(index),image)
        index += 1


##

list_label = os.listdir("data")

all_images = []
all_labels =  []
for label in list_label:
    list_images = os.listdir(os.path.join("data",label))
    
    

model.fit(

##
for epoch in range(config['n_epoch']):
    print("Epoch number --> ",epoch)
    for num_batch in range(100):
        images,labels = next(train_gen)
        images = np.stack(images,axis=0)
        labels = np.stack(labels,axis=0)
        loss,pred_loss,recons_loss,pred_acc,recons_acc = model.train_on_batch(images,{"prediction_output":labels,"reconstruction_output":images})
        print("Epoch {} batch {}".format(epoch,num_batch))
        print("Losses --> ",loss)
        print("prediction loss --> ",pred_loss)
        print("reconstruction loss --> ",recons_loss)
        print("prediction accuracy --> ",pred_acc)
        print("reconstruction accuracy--> ",recons_acc)
        print()
    
    loss_test = model.test_on_batch(test_image,{"prediction_output":test_labels,"reconstruction_output":test_image})
    print("Loss test --> ",loss_test)
  
##



