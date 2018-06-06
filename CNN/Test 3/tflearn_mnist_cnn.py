from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras import backend as K
from keras.utils import np_utils

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

import time

batch_size = 100
nb_classes = 10
epochs = 20

filters = 64

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

X_train, X_val = X_train[:-10000], X_train[-10000:]
Y_train, Y_val = Y_train[:-10000], Y_train[-10000:]


# TFLearn Network
network = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(network, filters, [3, 3], activation = 'relu', padding = 'same')

network = conv_2d(network, filters, [3, 3], activation = 'relu', padding = 'same')

network = fully_connected(network, 100, activation='relu')
network = dropout(network, 0.5)

network = fully_connected(network, 100, activation='relu')
network = dropout(network, 0.5)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer = 'adam', learning_rate = 0.01,
                     loss = 'categorical_crossentropy', name = 'target')

# Training
model = tflearn.DNN(network, tensorboard_verbose = 0)

start_time = time.time()

model.fit({'input': X_train}, {'target': Y_train}, n_epoch = epochs, validation_set = ({'input': X_val}, {'target': Y_val}), 
show_metric = True, batch_size = batch_size, shuffle = True, snapshot_epoch = True)

print("Total training time: %fs" % (time.time() - start_time))
print("Average time for an epoch: %fs" % ((time.time() - start_time) / epochs))

score = model.evaluate(X_test, Y_test)
print('Test score:', score)

