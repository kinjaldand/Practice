# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:10:12 2019

@author: A669593
"""


import matplotlib.pyplot as plt

def plothistory(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def evalAll(X_test, Y_test, VERBOSE,history,model,model_type):
    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print("Test score:", score[0])
    print('Test accuracy:', score[1])
    
    plothistory(history)
    
    print("TRAIN : ",history.history['acc'][-1])
    print("VAL : ",history.history['val_acc'][-1])
    print("TEST : ",score[1])    
    return (model_type,history.history['acc'][-1],history.history['val_acc'][-1],score[1])