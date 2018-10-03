import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import vtk
import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import Augmentor
from utils import load_config

path_config = "config.json"

config = load_config(path_config)
#SAVE IMAGES in .npy

os.chdir(r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire")

def give_generators(config):

    list_files = os.listdir(config['path_source'])
    all_images = []
    for file in list_files:
        #img = cv2.imread(os.path.join(path_imgs,file))
        #img = img[:,:,0]
        img = nib.load(os.path.join(config['path_source'],file))
        img = img.get_data()
        if img.shape == (440,440):
            all_images.append(img)
        else:
            print("Wrong shape for image {} here ->{}".format(file,img.shape))
        
    print("Initializing with {} pics".format(len(all_images)))
    
    df_labels = pd.read_csv(config['path_csv_labels'])
    df_labels.set_index("Unnamed: 0",inplace=True)
    df_labels.drop('sein_470',inplace=True)
    
    labels_possible =  list(df_labels['Type de lésion'].value_counts().keys())
    all_labels = df_labels['Type de lésion'].values
    all_encoded_labels = [labels_possible.index(k) for k in all_labels]
    
    train_images, test_images, train_labels, test_labels = train_test_split(all_images,all_encoded_labels,test_size=0.1)
    
    train_images = np.stack(train_images, axis=0)
    print("Train set shape for init --> {}".format(len(train_images)))
    print("Test set shape for init --> {}".format(len(test_images)))
    tr_labels = Augmentor.Pipeline.categorical_labels(train_labels)
    tes_labels = Augmentor.Pipeline.categorical_labels(test_labels)


    p = Augmentor.Pipeline()
    p.rotate(0.2, max_left_rotation=5, max_right_rotation=5)
    p.flip_top_bottom(0.5)
    p.flip_left_right(probability=0.5)
    p.random_distortion(probability=0.7, grid_width=4, grid_height=4, magnitude=8)    
    train_generator = p.keras_generator_from_array(train_images, tr_labels, batch_size=config['batch_size'])
    test_generator  = p.keras_generator_from_array(test_images, tes_labels, batch_size=config['batch_size'])
    return(train_generator,test_generator)

## TEST

#data_path= r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\data_raw"
#path_csv_labels = r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\train_set.csv"
#tr,tes = give_generators(config)
















