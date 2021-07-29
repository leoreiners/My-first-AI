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

#PT - transformando as imagens em matrizes EN - Converting images to arrays
def create_data(folder, name):
    for i in os.listdir(folder):
        image = Image.open(os.path.join(folder, i))
        image = Image.Image.resize(image, [200, 200])
        x = np.array(image)
        X.append(x)
        y.append(str_to_num[name])

create_data(gnome_folder, 'gnome')
create_data(drone_folder, 'drone')
