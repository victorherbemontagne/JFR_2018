import os
import numpy as np

import tensorflow as tf
from tensorflow import keras as K
import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import losses as losses
from tensorflow.python.keras.applications.vgg16 import VGG16 
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.models import Model 

#os.chdir(r"D:\Deepnews\deepnews_github\JFR_2018")
"""
File where we describe the loss that can be pluged to the model
"""


from utils import load_config
config = load_config()

## NOT USED

def add_loss(targett,output_net):
    to_feed_VGG = K.backend.concatenate([targett,output_net],axis=0)
    with tf.name_scope("loss"):
        with tf.name_scope("perceptual_loss"):
            perceptual_loss = add_perceptual_loss(to_feed_VGG)
            
        with tf.name_scope("pixel_loss"):
            pxl_loss = pixel_loss(targett,output_net)
            tf.summary.scalar("pixel_loss",pxl_loss)
    
        #assert perceptual_loss.get_shape() == pxl_loss.get_shape()
        
        loss = tf.add(pxl_loss,perceptual_loss)
        tf.summary.scalar("total_loss",loss)
    return(loss)
    
##

def loss_function(y_true,y_pred):
    pxl_loss = pixel_loss(y_true,y_pred)
    weighted_pxl_loss = kl.Lambda(lambda x: config["weight_pixel_loss"]*x)(pxl_loss)
    with tf.name_scope("perceptual_loss"):
        input_ = K.backend.concatenate([y_true,y_pred],axis=0)
        eval_vgg = K.backend.concatenate([input_,input_,input_],axis=-1)
        print("shape input vgg --> ",input_.shape)
        vgg = VGG16(weights='imagenet',
                    input_tensor=eval_vgg,
                    include_top=False)
        vgg.trainable = False
        layers = [l for l in vgg.layers]
 
        for layer in layers:
            layer.trainable = False
            if layer.name == "block3_conv3":
                eval_vgg = layer(eval_vgg)
                break
            eval_vgg = layer(eval_vgg)
        vgg_target = eval_vgg[:config["batch_size"]]
        vgg_prediction = eval_vgg[config["batch_size"]:]
        perceptual_loss = kl.Lambda(lambda x:config["weight_perceptual_loss"]*x)(pixel_loss(vgg_target,vgg_prediction))
    loss = kl.Add()([perceptual_loss,pxl_loss])
    return(loss)
    
def pixel_loss(y_true,y_pred):
    #tsize = tf.size(y_pred)
    #size = tf.to_float(tsize)
    with tf.name_scope('pixel_loss'):
        output = K.backend.mean(K.backend.square(y_pred - y_true),axis=[1,2,3])
    return output









