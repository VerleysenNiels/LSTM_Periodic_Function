#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:03:26 2019

@author: Niels Verleysen

Class that manages the basic LSTM model for prediction of the periodic function.
You first have to build the model by specifying the model architecture. Then you can train the model on given data.
Trained weights can then be used to predict values of the periodic function.
"""

from keras.models import Model
from keras.layers import LSTM, Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import losses
import numpy as np
import csv

class PeriodicFunctionLSTM:
    
    def __init__(self, architecture, k):
        inputs = Input(shape=(k,1))
        l = inputs
        for i in range(0, len(architecture)-1):
            l = LSTM(int(architecture[i]), return_sequences=True)(l)

        l = LSTM(int(architecture[-1]), return_sequences=False)(l)
        outputs = Dense(1)(l)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=losses.mean_absolute_error, optimizer='Adam', metrics=['accuracy', 'mae'])
        
        self.model.summary()
        
        filepath="./Weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]
        
    def train(self, X, Y, epochs, batch_size):
        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks_list)
        return history
    
    def load(self, file):
        self.model.load_weights(file)
    
    def predict(self, previous_vals):
        prediction = self.model.predict(previous_vals, verbose=0)
        return prediction

    """ 
        Own evaluation method
        Expects a set of input-output pairs and returns the average loss and the standard deviation of the loss
        It also outputs a csv file with all predictions and baselines, allowing us to make a plot afterwards
    """
    def evaluate(self, x, y, file):
        losses = []
        with open(file, 'w', newline='') as outfile:
            w = csv.writer(outfile)
            w.writerow(['Real', 'Predicted', 'Previous_measured', 'Mean_k_previous_measured'])
            xhat = []
            for i in range(0,len(x)):
                if len(xhat) == 0:
                    xhat = x[i]
                yhat = self.model.predict(np.expand_dims(xhat, axis=0))
                losses.append(abs(yhat - y[i]))
                w.writerow([y[i], yhat[0][0], x[i][-1][0], np.mean(x[i])])
                xhat = np.delete(xhat, 0)
                xhat = np.append(xhat, yhat)
                xhat = np.expand_dims(xhat, axis=1)

        losses = np.array(losses)
        return np.mean(losses), np.std(losses)

