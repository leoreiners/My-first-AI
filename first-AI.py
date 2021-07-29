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

#plot (opcional|optional)
    #plt.imshow(X[0])
    #plt.show()
    #print(X[0])

#PT - separando os dados entre treino e avaliação EN - Separating data between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#PT - criando os placeholders EN - Creating placeholders
x_place = tf.placeholder(tf.float32, shape = [None, 200, 200, 3])
y_place = tf.placeholder(tf.int32, shape = [None,])

#convertendo para uns e zeros
one_hot = tf.one_hot(y_place, 2)

#PT - transformando os arrays em tensors En - converting arrays to tensors
input_layer = tf.reshape(x_place, shape = [-1, 200, 200, 3])

#PT - criando os processos de aprendizado da rede neural EN - creating the learning process of the NN
flatten = tf.reshape(input_layer, [-1, (200*200*3)])
fc1 = tf.layers.dense(flatten, units=200, activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, units=200, activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2, units=200, activation=tf.nn.relu)
dropout = tf.layers.dropout(fc3, rate=0.2)
logits = tf.layers.dense(dropout, units=2)

#Pt - criando o loss e a velocidade de aprendizado com o loss EN - Creating loss anda learning speed with loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot))
optimiser = tf.train.AdamOptimizer()
training_op = optimiser.minimize(loss)

#PT - convertendo os resultados quebrados em exatos EN - Making all results integers
correct_pred = tf.equal(tf.argmax(one_hot, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#PT - Setando a quantidade de ciclos(EPOCHS) e o a quantidade de dados recebidos(BATCH_SIZE) EN- Setting up the number of cycles (EPOCHS) and the amount of data received (BATCH_SIZE)
EPOCHS = 40
BATCH_SIZE = 35

#PT - Etapa final, startando a IA - EN Final step, running the AI.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)

        for batch_start in range(0, len(X_train), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch_X, batch_y = X_train[batch_start:batch_end], y_train[batch_start:batch_end]

            sess.run(training_op, feed_dict={x_place:batch_X, y_place:batch_y})
            train_accuracy = sess.run(accuracy, feed_dict={x_place:X_train, y_place:y_train})
            test_accuracy = sess.run(accuracy, feed_dict={x_place:X_test, y_place:y_test})

            print('\nTeste: {}'.format(step))
            print('Imagens usadas: {} out of {}'.format(batch_start, len(X_train)))
            print('...')
            print('Precisão do treino: {a: 0.8f}'.format(a=train_accuracy))
            print('Precisão do teste: {a: 0.8f}'.format(a=test_accuracy))


