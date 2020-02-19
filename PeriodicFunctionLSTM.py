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
from keras.layers import LSTM, Input
from keras.callbacks import ModelCheckpoint
from keras import losses

class PeriodicFunctionLSTM:
    
    def __init__(self, architecture, k):
        inputs = Input(shape=(k,1))
        l = inputs
        for layer in architecture:
            l = LSTM(int(layer), return_sequences=True)(l)
            
        outputs = LSTM(1)(l)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=losses.mean_squared_error, optimizer='Adam', metrics=['accuracy', 'mae'])
        
        #self.model.summary()
        
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
    
    def evaluate(self, x, y):
        return self.model.evaluate(x=x, y=y)
