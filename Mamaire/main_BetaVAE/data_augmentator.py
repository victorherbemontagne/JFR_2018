import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import Augmentor

try:
    os.chdir("D:\Deepnews\deepnews_github\JFR_2018\Mamaire")
except Exception as e:
    print(e)
    pass

path_config = "config_for_outside.json"
from utils import load_config
config = load_config(path_config)
#SAVE IMAGES in .npy

def give_array_label(index,nbr_element,nbr_classe):
    """
    Output an array avec le label sous forme categorical
    """
    array = np.zeros((nbr_classe,nbr_element))
    array[index] = np.ones(nbr_element)
    return(np.transpose(array))
    


def give_generators(config):

    list_files = os.listdir(config['path_source'])
    all_images = []
    for file in list_files:
        img = nib.load(os.path.join(config['path_source'],file))
        img = img.get_data()
        if img.shape == (440,440):
            img = (img-np.mean(img))/np.std(img)
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
    labels = Augmentor.Pipeline.categorical_labels(all_encoded_labels)
    train_images, test_images, train_labels, test_labels = train_test_split(all_images,labels,test_size=0.1)
    
    train_images = np.stack(train_images, axis=0)
    print("Train set shape for init --> {}".format(len(train_images)))
    print("Test set shape for init --> {}".format(len(test_images)))

    label_to_image = {k:[] for k in range(17)}
    print('Building dictionnaries of all sources')
    for image,label in tqdm(zip(all_images,all_encoded_labels)):
        label_to_image[label].append(image) # pour créer un générateur par type d'image

    p = Augmentor.Pipeline()
    p.rotate(0.5, max_left_rotation=5, max_right_rotation=5)
    p.flip_top_bottom(0.7)
    p.flip_left_right(probability=0.5)
    p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=8) 
    print('Building generator dictionnary.. ')
    label_to_generator = {k: p.keras_generator_from_array(label_to_image[k],give_array_label(k,len(label_to_image[k]),17),
                                                            batch_size=config['nbr_image_by_batch']) for k in range(17)}
    #train_generator = p.keras_generator_from_array(train_images, train_labels, batch_size=config['batch_size'])
    #test_generator  = p.keras_generator_from_array(test_images, test_labels, batch_size=config['batch_size'])
    return(label_to_generator)

##




## TEST

try:
    #data_path= r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\data_raw"
    #path_csv_labels = r"D:\Deepnews\deepnews_github\JFR_2018\Mamaire\train_set.csv"
    #tr,tes = give_generators(config)
    assert False
    index = 0
    les_labels = {k:0 for k in range(17)}
    for img,labels in tqdm(tes):
        if index < 100:
            for label in labels:
                index_label = list(label).index(1)
                les_labels[index_label] += 1
            index += 1
        else:
            break
    
    
    #for images,labels in tes:
    #    for image in images:
    #        print(image)
    #        print(image.shape)
    #        break
    #    break
    
    pipelines = {}
    for folder in folders:
        print("Folder %s:" % (folder))
        pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
        print("\n----------------------------\n")
except Exception as e:
    pass







