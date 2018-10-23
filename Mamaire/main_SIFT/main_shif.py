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




##

input_shape = [500,32]
