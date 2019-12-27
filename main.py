#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:16:18 2019

@author: Niels Verleysen

Main script for this experiment.

The basic function is building a LSTM model and training it on a given periodic function with/without noise, 
with a given sampling frequency and amount of previous values given to the model (k). The accuracy of this model is then tested.

This should then be done for varying frequencies, varying k and varying noise of the periodic function.
"""

from PeriodicFunctionLSTM import PeriodicFunctionLSTM
from PeriodicFunction import PeriodicFunction
from random import randint
from random import seed
import numpy as np

"""Define global variables"""
EPOCHS = 15
BATCH_SIZE = 20
DATASET_SIZE = 200
TESTSET_SIZE = 50
SEED = 36947

"""
Build a training or testing dataset with values from given function with given sampling rate.
Group k previous true values as next input for the model.
"""
def build_dataset(function, k, sampling_rate, test=False):
    values = []
    X = []
    Y = []
    size = TESTSET_SIZE if test else DATASET_SIZE
    startpoint = randint(0,100)
    
    #Generate values
    for i in range(startpoint, size + k + startpoint):
        values.append(function.value(sampling_rate * i))
    
    #Use generated values to build the training data
    for i in range(0, size):
        index = i + k  # current value for Y; we start from element k in values list
        Y.append(values[index])
        x = []  #List with k previous values
        for j in range (index - k, index):
            x.append(np.array([values[j-1]]))
        X.append(np.array(x))
    
    return np.array(X), np.array(Y)

"""
Run basic experiment:
    Build the model with given specifications.
    Use build_dataset function to build the training data.
    Train the model on this dataset.
    Test the accuracy of the model.
"""
def run(function, architecture, k, sampling_rate):
    model = PeriodicFunctionLSTM(architecture,k)
    X,Y = build_dataset(function, k, sampling_rate)
    model.train(X, Y, EPOCHS, BATCH_SIZE)
    X,Y = build_dataset(function, k, sampling_rate, test=True)
    return model.evaluate(X, Y)

"""
MAIN
Use run function to test model on function with different types of noise, different k values and different sampling rates.
Collect the accuracies from these tests.
"""
if __name__ == '__main__':
    seed(SEED)
    
    #test working of one run
    function = PeriodicFunction(35, 2)
    function.add_gaussian_noise(4)
    
    k = 5
    sampling = 3
    
    result = run(function, [5], k, sampling)
    
    print(result)
