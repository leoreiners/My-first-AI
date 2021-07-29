import os
import random
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#PT - criar listas vazias EN - Create empty lists
X = []
y = []

str_to_num = {'gnome':0, 'drone':1}

#PT - criando as variáveis dos diretórios EN - creating directory variables
gnome_folder = 'loading_data/gnome'
drone_folder = 'loading_data/drone'
