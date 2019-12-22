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

from LSTM import PeriodicFunctionLSTM

"""Define global variables"""
EPOCHS = 15
BATCH_SIZE = 20

"""
Build a training dataset with values from given function with given sampling rate.
Group k previous true values as next input for the model.
"""
def build_dataset(function, k, sampling_rate):
    print("ToDo")

"""
Run basic experiment:
    Build the model with given specifications.
    Use build_dataset function to build the dataset.
    Train the model on this dataset.
    Test the accuracy of the model.
"""
def run(function, architecture, k, sampling_rate):
    model = PeriodicFunctionLSTM(architecture,[k])
    dataset = build_dataset(function, k, sampling_rate)
    model.train(dataset[0], dataset[1], EPOCHS, BATCH_SIZE)
    
    #ToDo: test model and return accuracy

"""
MAIN
Use run function to test model on function with different types of noise, different k values and different sampling rates.
Collect the accuracies from these tests.
"""
if __name__ == '__main__':
    print("ToDo")