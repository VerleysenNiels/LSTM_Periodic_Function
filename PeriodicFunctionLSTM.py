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
from keras.layers import LSTM, Input, Dense#, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import losses
import numpy as np
import csv

class PeriodicFunctionLSTM:
    
    def __init__(self, architecture_LSTM, k, architecture_CNN = [], architecture_FC = [], m=1):
        inputs = Input(shape=(k,1))
        l = inputs

        #if len(architecture_CNN) > 0:
            #for layer in architecture_CNN:
                #l = Conv1D(layer[0], layer[1])(l)
                #l = MaxPooling1D()(l)

        for i in range(0, len(architecture_LSTM)-1):
            l = LSTM(int(architecture_LSTM[i]), return_sequences=True)(l)  #activation='sigmoid',

        l = LSTM(int(architecture_LSTM[-1]), return_sequences=False)(l)

        if len(architecture_FC) > 0:
            for layer in architecture_FC:
                l = Dense(layer)(l)

        outputs = Dense(m)(l)
        
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
        Own evaluation method (for predicting one step in the future and then using this prediction as part of the input for the next prediction)
        Expects a set of input-output pairs and returns the average loss and the standard deviation of the loss
        It also outputs a csv file with all predictions and baselines, allowing us to make a plot afterwards
    """
    def evaluate_m1(self, x, y, file):
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

    """ 
        Same evaluation as before, but changed to allow multi-step predictions
        Expects a set of input-output pairs and returns the average loss and the standard deviation of the loss
        It also outputs a csv file with all predictions, allowing us to make a plot afterwards
    """
    def evaluate_multi_step(self, x, y, file, m, k):
        losses = []
        with open(file, 'w', newline='') as outfile:
            w = csv.writer(outfile)
            w.writerow(['Real', 'Predicted', 'Previous_measured', 'Mean_k_previous_measured'])
            xhat = x[0]
            for i in range(0, round(len(y)/m)):
                yhat = self.model.predict(np.expand_dims(xhat, axis=0))
                for j in range(0, m):
                    losses.append(abs(yhat[0][j] - y[i*m +j]))
                    w.writerow([y[i*m +j][0], yhat[0][j], x[i][-1][0], np.mean(x[i])])
                if k > m:
                    for j in range(0, m):
                        xhat = np.delete(xhat, 0)
                    xhat = np.append(xhat, yhat)
                else:
                    diff = m - k
                    xhat = yhat[diff:-1]
                xhat = np.expand_dims(xhat, axis=1)

        losses = np.array(losses)
        return np.mean(losses), np.std(losses)

    """
        Predict the next M values given the initial k values as Start
    """
    def predict_next_M(self, Start, M):
        predictions = []
        xhat = []
        for i in range(0, M):
            if len(xhat) == 0:
                xhat = Start
            yhat = self.model.predict(np.expand_dims(xhat, axis=0))
            predictions.append(yhat[0][0])
            xhat = np.delete(xhat, 0)
            xhat = np.append(xhat, yhat)
            xhat = np.expand_dims(xhat, axis=1)
        return predictions
