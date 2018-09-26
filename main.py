"""
Script that were used to test the implementation in Tensorflow

/!\ /!\ /!\
It's not the model I choosed to use in production // SEE ACTUAL SCRIPT TO USE.py
/!\ /!\ /!\


"""

import os
import cv2
import time
import itertools
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import losses as losses
from tensorflow.python.keras.applications.vgg16 import VGG16 
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image as img

os.chdir(r"D:\Deepnews\deepnews_github\JFR_2018")

from utils_model import encode,decode,residual_block, build_net

from losses import loss_function
from data_generator import Generator_img

#MODEL DEFINITION

    

from utils import load_config
    
config = load_config()



tf.reset_default_graph()



#Start building

inputt = K.backend.placeholder(shape=[config["batch_size"]]+config["input_shape"],name = "input_tensor")
targett = K.backend.placeholder(shape=[config["batch_size"]]+config["input_shape"],name = "target_tensor")

output_net = build_net(config, inputt)

loss_batch = loss_function(targett,output_net)
mean_cost = tf.reduce_mean(loss_batch)

with tf.name_scope("train_ops"):
    optimizer = tf.train.AdamOptimizer(config["learning_rate"])
    training_op = optimizer.minimize(mean_cost)
    

init_1 = tf.global_variables_initializer()
init_2 = tf.local_variables_initializer()

for variable in tf.trainable_variables():
    tf.summary.histogram(variable.name,variable)
tf.summary.scalar("Loss",mean_cost)
tf.summary.image("input",inputt)
tf.summary.image("output_of_net",output_net)
tf.summary.image("target",targett)
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    try:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("summaries/test_5")
        writer.add_graph(sess.graph)
        
        K.backend.set_session(sess)
        
        init_1.run()
        init_2.run()
        
        data = Generator_img(config)
        gen = data.generator
        epoch_prev = -1
        num_pass = -1
        for epoch,batch_glasses,batch_no_glasses in gen:
            
            start = time.time()
                
            _,cost_int,summary = sess.run([training_op,mean_cost,merged_summary],feed_dict={
                                                                        "input_tensor:0": batch_glasses,
                                                                        "target_tensor:0":batch_no_glasses
                                                                        })
            num_pass += 1
            writer.add_summary(summary,num_pass)
            print("Epoch -->{}, batch -->{}, loss -->{}".format(epoch,num_pass,cost_int))
            #test_images = make_test(config["path_test_set"],sess,config["batch_size"])
            
            #tf.summary.image("images",test_images,3)
            
            
            if epoch != epoch_prev:
                print("Epoch number --> ",epoch)
                duration_epoch = time.time() - start
                print("End epoch {} in {}s".format(epoch_prev,duration_epoch))
                save_path = saver.save(sess,"model_saved/model.ckpt")
                print("Model saved in --> {}".format(save_path))
                print()
                epoch_prev = epoch
    except Exception as e:
        sess.close()
        raise e




