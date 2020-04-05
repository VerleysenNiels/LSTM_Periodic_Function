#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 4 14:03:11 2020

@author: Niels Verleysen

Test a trained model on a periodic signal with multiple frequencies on the same signal where one of the higher frequency components decays over time.
The model always predicts the next two periods of the signal based on the previous K values. This is done for each point of the signal.
Results are saved in a csv file where the predictions from each point onwards are put in a different column.
This csv file can then be used to plot all predictions for each additional point.
These plots are then used as frames in a video to show how the model reacts to the decaying high frequency component.
"""

import os
import pickle
from PeriodicFunctionLSTM import PeriodicFunctionLSTM
from PeriodicFunction import PeriodicFunction
from random import randint
from random import seed
import numpy as np
import csv


"""Define global variables"""
SIZE = 700  # Amount of timesteps tested
M = 120      # Amount of steps predicted in the future (here 2 periods of 60 minutes)
SEED = 36947

"""
Initialize a 2D array with the rows for the csv file
"""
def init_output_array(Y, k):
    output_array = np.expand_dims(Y, axis=1)
    header_list = ["Real Signal"]
    for i in range(k, SIZE-1):
        header_list.append("Prediction from step: " + str(i))
    return header_list, output_array.tolist()


"""
Set up the dataset
"""
def build_dataset(function, k, sampling_rate):
    values = []
    X = []
    Y = []
    startpoint = randint(0,100)
    
    #Generate values
    for i in range(startpoint, SIZE + startpoint + k):
        values.append(function.value(sampling_rate * i))
    
    #Use generated values to build the test data
    for i in range(0, SIZE):
        index = i + k  # current value for Y; we start from element k in values list
        Y.append(values[index])
        x = []  #List with k previous values
        for j in range(index - k, index):
            x.append(np.array([values[j-1]]))
        X.append(np.array(x))
    
    return np.array(X), np.array(Y), values

"""
MAIN
Use the model to predict the next M values from each point in the dataset, write output to a csv file
"""
if __name__ == '__main__':
    seed(SEED)

    """Init signal"""
    test_function = PeriodicFunction(10, 0.016667)
    test_function.add_disturbing_decaying_amp(0.01, 4, 60, 0.4)
    
    """Determine sampling rate and k to use"""
    sampling_rate = 1  # 0.1 second: 0.0016667, 1 second: 0.016667, 1 minute: 1
    k = 16

    """Load model"""
    model = PeriodicFunctionLSTM([128, 64, 32], k)
    model.load("./Trained/Dense as output/Decaying_0,01_A_4_F_0,4/Network_128_64_32/sample_1_k_32")

    """Init dataset"""
    X, Y, Values = build_dataset(test_function, k, sampling_rate)

    """Init output array"""
    header_array, output_array = init_output_array(Values, k)

    """Predict next M values from each point (after timestep k)"""
    for i in range(0, len(X)):
        print("Predicting from point: " + str(i))
        predictions = model.predict_next_M(X[i], M)
        print("Writing " + str(SIZE - i - 1) + " points:")
        print(predictions)
        for j in range(i+k+1, SIZE+k):
            if j <= i+k+M:
                output_array[j].append(predictions[j-i-k-1])
            else:
                output_array[j].append(0)
            
    
    """Write results to a csv file"""
    with open('./Video/Model_128_64_32_K_16/Results.csv', 'w', newline='') as outfile:
        w = csv.writer(outfile)
        w.writerow(header_array)
        for row in output_array:
            w.writerow(row)
