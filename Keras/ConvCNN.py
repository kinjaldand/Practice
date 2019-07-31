# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:15:57 2019

@author: A669593
"""


from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D, Activation, Dropout,Flatten,Dense

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import fashion_mnist
from keras.optimizers import RMSprop, Adam
from keras import regularizers

import AllKerasFunctions as akf
np.random.seed(3007)  # Reproducibility


results=[]

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# network and training
NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = SGD() # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3
IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
NB_CLASSES = 10 # number of outputs = number of digits
INPUT_SHAPE = (IMG_ROWS, IMG_COLS,1)


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



x_train = x_train/255
x_test = x_test/255

x_train = x_train[:,  :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
print(x_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)


model = Sequential()
model.add(Convolution2D(10, kernel_size=5, padding="same",
input_shape=INPUT_SHAPE))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(20, kernel_size=5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation("relu"))
# a softmax classifier
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))

model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])
history = model.fit(x_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

res=akf.evalAll(x_test, Y_test, VERBOSE,history,model,"CNN1")
results.append(res)
