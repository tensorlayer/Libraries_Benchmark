from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

import time

nb_classes = 10
neurons = 800
epochs = 200
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

X_train, X_val = X_train[:-10000], X_train[-10000:]
Y_train, Y_val = Y_train[:-10000], Y_train[-10000:]

print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')
print(X_test.shape[0], 'test samples')


print(Y_train.shape, Y_test.shape)

# TFLearn Network
network = input_data(shape=[None, 784], name='input')

network = fully_connected(network, neurons, activation='relu')
network = dropout(network, 0.5)

network = fully_connected(network, neurons, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer = 'adam', learning_rate = 0.01,
                     loss = 'categorical_crossentropy', name = 'target')

# Training
model = tflearn.DNN(network, tensorboard_verbose = 0)

start_time = time.time()

model.fit({'input': X_train}, {'target': Y_train}, n_epoch = epochs, validation_set = ({'input': X_val}, {'target': Y_val}), 
show_metric = True, batch_size = 500, shuffle = True, snapshot_epoch = True)

print("Total training time: %fs" % (time.time() - start_time))
print("Average time for an epoch: %fs" % ((time.time() - start_time) / epochs))

score = model.evaluate(X_test, Y_test)
print('Test score:', score)

