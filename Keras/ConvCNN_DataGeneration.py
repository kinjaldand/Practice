# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:30:42 2019

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

# augumenting
print("Augmenting training set images...")
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

"""import numpy as np
NUM_TO_AUGMENT=5

xtas, ytas = [], []
for i in range(x_train.shape[0]):
    num_aug = 0
    x = x_train[i] # (3, 32, 32)
    x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
    for x_aug in datagen.flow(x, batch_size=1,save_to_dir='preview', save_prefix='cifar', save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug += 1

"""

#fit the dataget
datagen.fit(x_train)


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
model.add(Dropout(0.2))
# a softmax classifier
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))

model.summary()
model.compile(loss='categorical_crossentropy',
optimizer=OPTIMIZER,
metrics=['accuracy'])
history = model.fit_generator(datagen.flow(x_train, Y_train,
batch_size=BATCH_SIZE), samples_per_epoch=x_train.shape[0],
epochs=NB_EPOCH, verbose=VERBOSE)

res=akf.evalAll(x_test, Y_test, VERBOSE,history,model,"CNN1")
results.append(res)
