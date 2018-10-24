from tensorflow.python.keras import layers as kl
from tensorflow.python.keras.models import Model
import tensorflow as tf	
from tensorflow.python.keras import backend as K
from tensorflow.python import keras as kers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.activations import selu


initializer = kers.initializers.he_normal()


def residual_block(x,index,step="encoding"):
    """
    Code for residual block
    """
    if step != "encoding":
        index = "decode_{}".format(index)
    
    x_new = kl.Conv2D(128,(3,3),padding="same",activation="selu",kernel_initializer=initializer,
                            name="conv1_res_{}".format(index))(x)
    
    x_new = kl.Conv2D(128,(3,3),padding="same",kernel_initializer=initializer,
                    name="conv2_res_{}".format(index))(x_new)
    
    x_out = kl.Add()([x,x_new])
    x_out = kl.Activation("relu")(x_out)
    return(x_out)

def encode(input_tensor):
    
    x = kl.Conv2D(36,5,strides=(2,2),activation="selu",kernel_initializer=initializer,
                    padding="same",name="conv_2")(input_tensor)
    
    x = kl.Conv2D(128,3,strides=(2,2),activation="selu",kernel_initializer=initializer,
                    padding="same",name="conv_3")(x)
    
    x = kl.Conv2D(128,3,strides=(2,2),activation="selu",kernel_initializer=initializer,
                    padding="same",name="conv_4")(x)
    return(x)

def decode(x):
    
    x = kl.Conv2DTranspose(128,3,strides=(2,2),padding="same",kernel_initializer=initializer,
                            activation="selu",name="deconv_1")(x)
    x = kl.BatchNormalization(name="batchnorm_decode_1")(x)
    #print("start decoding shape ",x.shape)
    
    x = kl.Conv2DTranspose(36,3,strides=(2,2),padding="same",kernel_initializer=initializer,
                            activation="selu",name="deconv_2")(x)
    x = kl.BatchNormalization(name="batchnorm_decode_2")(x)	
    #print("Mid decoding shape ",x.shape)
    
    x = kl.Conv2DTranspose(1,5,strides=(2,2),padding="same",kernel_initializer=initializer,
                            activation="selu",name="reconstruction_output")(x) 
    return(x)
    

def build_net(config,input_tensor):
    with tf.name_scope("base_network"):
        with tf.name_scope("encoding_network"):
            x = encode(input_tensor)
        with tf.name_scope('coding'):
            for k in range(config['nbr_residuals']):
                x = residual_block(x,k)
            pre_feature_vector = kl.Flatten(name="pre_feature_computation")(x)
            feature_vector = kl.Dense(100,activation="selu", name="feature_vector")(pre_feature_vector)
            post_feature_vector = kl.Dense(pre_feature_vector.shape[1],activation="selu",kernel_initializer=initializer, name="post_feature_vector")(feature_vector)
            x = kl.Reshape(x.shape[1:])(post_feature_vector)
            for k in range(config['nbr_residuals']):
                x = residual_block(x,k,step="decoding")
        with tf.name_scope('decoding'):
            output = decode(x)
    return(feature_vector,output)

    
    
    
    