# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:53:20 2019

@author: A669593
"""
from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import fashion_mnist
from keras.optimizers import RMSprop, Adam
from keras import regularizers

import AllKerasFunctions as akf
np.random.seed(3007)  # Reproducibility


results=[]


# network and training
NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = SGD() # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
RESHAPED = 784

X_train = x_train.reshape(60000, RESHAPED)
X_test = x_test.reshape(10000, RESHAPED)
X_train = y_train.astype('float32')
X_test = y_test.astype('float32')
# normalize
X_train = X_train/255
X_test = X_test/255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)


model=Sequential()
model.add(Dense(NB_CLASSES,input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])

history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

res=akf.evalAll(X_test, Y_test, VERBOSE,history,model,"Single_Layer")
results.append(res)
"""
TRAIN :  0.8635625
VAL :  0.8563333331743876
TEST :  0.8432
"""


model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])
history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

res=akf.evalAll(X_test, Y_test, VERBOSE,history,model,"Multi_Layer")
results.append(res)

"""
TRAIN :  0.9121458333333333
VAL :  0.8849166665077209
TEST :  0.8794
"""

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])
history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

res=akf.evalAll(X_test, Y_test, VERBOSE,history,model,"Dropout")
results.append(res)

"""

TRAIN :  0.8832291666666666
VAL :  0.8832500001589457
TEST :  0.8752
"""

OPTIMIZER = RMSprop() # optimizer
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])
history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

res=akf.evalAll(X_test, Y_test, VERBOSE,history,model,"RMSprop")
results.append(res)

OPTIMIZER = Adam() # optimizer
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])
history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

res=akf.evalAll(X_test, Y_test, VERBOSE,history,model,"Adam")
results.append(res)

OPTIMIZER = Adam() # optimizer
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,) , kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN , kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES , kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])
history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

res=akf.evalAll(X_test, Y_test, VERBOSE,history,model,"Regularizer")
results.append(res)
