from tensorflow.python.keras import layers as kl
from tensorflow.python.keras.models import Model
import tensorflow as tf	
from tensorflow.python.keras import backend as K


def residual_block(x,index,step="encoding"):
    """
    Code for residual block
    """
    if step != "encoding":
        index = "decode_{}".format(index)
    
    x_new = kl.Conv2D(128,(3,3),padding="same",activation="relu",name="conv1_res_{}".format(index))(x)
    x_new = kl.BatchNormalization(name="batchnorm_1_res_{}".format(index))(x_new)
    
    x_new = kl.Conv2D(128,(3,3),padding="same",name="conv2_res_{}".format(index))(x_new)
    x_new = kl.BatchNormalization(name="batchnorm_2_res_{}".format(index))(x_new)
    
    x_out = kl.Add()([x,x_new])
    x_out = kl.Activation("relu")(x_out)
    return(x_out)

def encode(input_tensor):
    x = kl.Conv2D(18,9,strides=(2,2),activation="relu",padding="same",name="conv_1")(input_tensor)
    x = kl.BatchNormalization(name="batchnorm_encode_1")(x)
    
    x = kl.Conv2D(36,3,strides=(2,2),activation="relu",padding="same",name="conv_2")(x)
    x = kl.BatchNormalization(name="batchnorm_encode_2")(x)
    
    x = kl.Conv2D(128,3,strides=(2,2),activation="relu",padding="same",name="conv_3")(x)
    x = kl.BatchNormalization(name="batchnorm_encode_3")(x)
    return(x)

def decode(x):
    
    x = kl.Conv2DTranspose(128,3,strides=(2,2),padding="same",activation="relu",name="deconv_1")(x)
    x = kl.BatchNormalization(name="batchnorm_decode_1")(x)
    print("start decoding shape ",x.shape)
    
    x = kl.Conv2DTranspose(36,3,strides=(2,2),padding="same",activation="relu",name="deconv_2")(x)
    x = kl.BatchNormalization(name="batchnorm_decode_2")(x)	
    print("Mid decoding shape ",x.shape)
    
    x = kl.Conv2DTranspose(1,9,strides=(2,2),padding="same",name="deconv_3")(x)
    x = kl.BatchNormalization(name="batchnorm_decode_3")(x)	
    print("End decoding shape ",x.shape) 
    return(x)
    

def build_net(config,input_tensor):
    with tf.name_scope("base_network"):
        x = encode(input_tensor)
        for k in range(config['nbr_residuals']):
            x = residual_block(x,k)
        pre_feature_vector = kl.Flatten(name="pre_feature_computation")(x)
        feature_vector = kl.Dense(128, name="feature_vector")(pre_feature_vector)
        output= kl.Dense(17,activation="softmax",name="post_feature_computation")(feature_vector)
        print("x shape ",x.shape)
#        x = kl.Reshape(x.shape[1:])(x)
        print(x.shape)
#        for k in range(config['nbr_residuals']):
#            x = residual_block(x,k,step="decoding")
        #output = decode(x)
    return(output)
    
    
    
    
    
    
    
    
    
    