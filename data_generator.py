"""
Build a data generator for the perceptual problem

test can be run in the part bellow the main code

"""

import os
import cv2
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm



from tensorflow.python.keras.preprocessing import image as img
os.chdir(r"D:\Deepnews\deepnews_github\JFR_2018")

from utils import load_config

config = load_config()

class Generator_img():

	def __init__(self,config,training=True):
		if training:
			self.path_source = config['path_source']
			self.path_masks  = config['path_mask']
		else:
			self.path_source = config['path_test_source']
			self.path_masks  = config['path_test_mask']
		
		self.list_sources = os.listdir(self.path_source)
		self.list_masks = os.listdir(self.path_masks)
		self.batch_size = config["batch_size"]
		self.n_epoch = config["n_epoch"]
		self.prepare_data()
		
		self.generator = self.generator(self.all_couples,self.batch_size,self.n_epoch,training=training)


	def get_list_path(self,path):
		list_glasses = os.listdir(path)
		all_paths = []	
		for type_glasses in list_glasses:
			path_type = os.path.join(path,type_glasses)
			glasses_by_type = [os.path.join(path_type,path) for path in os.listdir(path_type)]
			all_paths += glasses_by_type
		return(all_paths)

	def prepare_data(self):
		"""
		Function in charge of opening each image, creating the masked_images
		Populate la list self.couples qui sert juste à faire les couples images, masked_image
		"""
		self.all_couples = []
		for file_source,file_mask in tqdm(zip(self.list_sources,self.list_masks)):
			source = nib.load(os.path.join(self.path_source,file_source))
			source = source.get_data().astype(np.int16)
		
			mask = nib.load(os.path.join(self.path_masks,file_mask))    
			mask = mask.get_data().astype(np.int16)
			masked_image = mask * source
			self.all_couples.append([source,masked_image])

#self.generator(couples,self.batch_size,self.n_epoch,self.path_glasses,self.path_no_glasses)

	def generator(self,list_elmts,batch_size,n_epoch,training):
		#Lui va yield image_without_glasses et paired image_with_glasses
		if training:
			epoch_number = -1
			print("Nbr training examples -->",len(list_elmts))
			print("nbr iteration by epoch -->",len(list_elmts)//batch_size)
			while True:
				epoch_number += 1
				for k in range(len(list_elmts)//batch_size):
					batch = random.sample(list_elmts,batch_size)
					batch_source = np.array([elmt[0] for elmt in batch])
					batch_masks = np.array([elmt[1] for elmt in batch])
					yield(epoch_number,batch_source,batch_masks)
		else:
			while True:
				print(len(list_elmts))
				print(len(list_elmts)//batch_size)
				batch_source = np.array([elmt[0] for elmt in list_elmts])
				batch_masks = np.array([elmt[1] for elmt in list_elmts])
				yield(batch_source,batch_masks)

	@staticmethod
	def shuffle(list_elmts):
		return(np.random.shuffle(list_elmts))
		
		
		

## TEST 

#data_generator = Generator_glasses(config, training = True)

#gene = data_generator.generator


#for batch_1,batch_2 in gene:
#	print(len(batch_1))
#	print(len(batch_2))
#	break


#for index,pic in enumerate(batch_2):
#	cv2.imwrite("test/{}.jpg".format(index),pic)

#def plot(pic):
#	cv2.imshow("win",pic)
#	cv2.waitKey()



		