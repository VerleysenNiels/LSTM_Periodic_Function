#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:16:18 2019

@author: Niels Verleysen

Main script for this experiment.

The basic function is building a LSTM model and training it on a given periodic function with/without noise, 
with a given sampling frequency and amount of previous values given to the model (k). The accuracy and absolute error of this model are then tested.

This should then be done for varying frequencies, varying k and varying noise of the periodic function.
"""

import os
import pickle
from PeriodicFunctionLSTM import PeriodicFunctionLSTM
from PeriodicFunction import PeriodicFunction
from random import randint
from random import seed
import numpy as np
import csv

"""Models are too small, single cpu is faster"""
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""Define global variables"""
EPOCHS = 35
BATCH_SIZE = 200
DATASET_SIZE = 200000
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
def run(training_function, test_function, architecture, k, sampling_rate):
    model = PeriodicFunctionLSTM(architecture,k)
    X,Y = build_dataset(training_function, k, sampling_rate)
    history = model.train(X, Y, EPOCHS, BATCH_SIZE)
    X,Y = build_dataset(test_function, k, sampling_rate, test=True)
    eval = model.evaluate(X, Y)
    return eval, history

"""
MAIN
Use run function to test model on function with different types of noise, different k values and different sampling rates.
Collect the accuracies from these tests.
"""
if __name__ == '__main__':
    seed(SEED)
    
    """Determine function"""
    training_function = PeriodicFunction(10, 0.016667)
    training_function.add_disturbingf_increasing_freq(0.01, 0.001, 3)

    test_function = PeriodicFunction(10, 0.016667)
    
    """Determine different sampling rates to use"""
    sampling_rates = [0.0016667, 0.016667, 1, 15, 30, 55] # 0.1 second, 1 second, 1 minute, 15 minutes, ...
    
    """Do experiment for each sampling rate on the function; search over different k-values"""
    results = []
    for sampling_rate in sampling_rates:
        for i in range(0,6):
            k = 2**i
            result, history = run(training_function, test_function, [10, 10], k, sampling_rate)
            with open("./Results/Training/sample_" + str(sampling_rate) + "_k_" + str(k), 'wb') as outfile:
                pickle.dump(history, outfile)
            templist = [sampling_rate, k]
            templist.extend(result)            
            results.append(templist)
            
    
    """Write results dictionary as a table to a csv file"""
    with open('./Results/table.csv', 'w') as outfile:
        w = csv.writer(outfile)
        w.writerow(['Sampling rate', 'k', 'mse', 'accuracy', 'mae'])
        for row in results:
            w.writerow(row)
