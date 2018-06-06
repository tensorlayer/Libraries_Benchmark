from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from keras import backend as K
from keras.utils import np_utils

import time

batch_size = 100
nb_classes = 10
epochs = 20

filters = 256

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



model = Sequential()
model.add(Conv2D(filters, kernel_size=(3, 3),
                 activation = 'relu',
                 input_shape = input_shape, padding = 'same'))
model.add(Conv2D(filters, (3, 3), activation='relu', padding = 'same'))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['accuracy'])

start_time = time.time()
history = model.fit(X_train, Y_train,
                    batch_size = batch_size, epochs = epochs,
                    verbose = 1, validation_data = (X_val, Y_val)) 
print("Total training time: %fs" % (time.time() - start_time))
print("Average time for an epoch: %fs" % ((time.time() - start_time) / epochs))
score = model.evaluate(X_test, Y_test, verbose = 0)
print('Test score:', score[0])
print('Test accuracy:', score[1])