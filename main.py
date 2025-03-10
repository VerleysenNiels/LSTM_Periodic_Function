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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Uncomment for small models, as a single cpu is faster in that case
"""
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""Define global variables"""
EPOCHS = 50
BATCH_SIZE = 200
DATASET_SIZE = 200000
TESTSET_SIZE = 700
SEED = 36947

"""
Build a training or testing dataset with values from given function with given sampling rate.
Group k previous true values as next input for the model.
"""
def build_dataset(function, k, sampling_rate, m=1, test=False):
    values = []
    X = []
    Y = []
    size = TESTSET_SIZE if test else DATASET_SIZE
    startpoint = 0 #randint(0,100)
    
    #Generate values
    for i in range(startpoint, size + k + startpoint + m):
        values.append(function.value(sampling_rate * i))
    
    #Use generated values to build the training data
    for i in range(0, size):
        index = i + k  # current value for Y; we start from element k in values list
        y = []
        for a in range(0, m):
            y.append(values[index+a])
        Y.append(y)
        x = []  #List with k previous values
        for j in range(index - k, index):
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
def run(training_function, test_function, architecture_LSTM, k, sampling_rate, architecture_CNN=[], architecture_FC=[], m=1):
    model = PeriodicFunctionLSTM(architecture_LSTM, k, architecture_CNN=architecture_CNN, architecture_FC=architecture_FC, m=m)
    X,Y = build_dataset(training_function, k, sampling_rate, m=m)
    history = model.train(X, Y, EPOCHS, BATCH_SIZE)
    model.model.save("./Trained/sample_" + str(sampling_rate) + "_k_" + str(k))
    X,Y = build_dataset(test_function, k, sampling_rate, test=True)
    eval = model.evaluate_multi_step(X, Y, "./Predictions/sample_" + str(sampling_rate) + "_k_" + str(k) + ".csv", m, k)
    #eval = model.evaluate_m1(X, Y, "./Predictions/sample_" + str(sampling_rate) + "_k_" + str(k) + ".csv")
    return eval, history

"""
Calculate absolute difference of given prediction
"""
def absolute_difference(real, predicted):
    difference = 0
    for i in range(0, len(predicted)):
        difference += abs(real - predicted[i])
    return difference

"""
Calculate mean absolute difference of given prediction
"""
def mean_absolute_difference(real, predicted):
    difference = 0
    for i in range(0, len(predicted)):
        difference += abs(real - predicted[i])
    return difference/len(predicted)

"""
Voting based anomaly detection
"""
def vote(real, predicted, voting_threshold):
    votes = 0
    for i in range(0, len(predicted)):
        if voting_threshold < abs(real - predicted[i]):
            votes += 1
    return votes

"""
Use run function to test model on function with different types of noise, different k values and different sampling rates.
Collect the accuracies from these tests.
"""
def train_and_test():
    """Determine function"""
    training_function = PeriodicFunction(10, 0.016667)
    training_function.add_disturbing_signal(4, 0.4)
    #training_function.add_gaussian_noise(0.2)

    test_function = PeriodicFunction(10, 0.016667)
    test_function.add_disturbing_signal(4, 0.4)

    """Determine different sampling rates to use"""
    sampling_rates = [1]  # 0.1 second: 0.0016667, 1 second: 0.016667, 1 minute

    """Do experiment for each sampling rate on the function"""
    results = []
    for sampling_rate in sampling_rates:
        for k in [32, 64, 128]:
            result, history = run(training_function, test_function, [200, 200], k, sampling_rate)  # , architecture_FC=[200, 200], m=30)  #, architecture_FC=[200, 200])  #architecture_CNN=[[128, 6], [64, 5], [32, 4]]
            with open("./Results/Training_history/sample_" + str(sampling_rate) + "_k_" + str(k), 'wb') as outfile:
                pickle.dump(history, outfile)
            templist = [sampling_rate, k]
            templist.extend(result)
            results.append(templist)

    """Write results dictionary as a table to a csv file"""
    with open('./Results/table.csv', 'w', newline='') as outfile:
        w = csv.writer(outfile)
        w.writerow(['Sampling rate', 'k', 'mae', 'stdv'])
        for row in results:
            w.writerow(row)

"""
Test anomaly detection capabilities
"""
def test_anomaly_detection():

    # Tuning variables
    start = 228     # When does the anomaly start
    name = 'FreqDev'  # Output file name
    titlerow = ['Actual']

    # Load trained model
    model = PeriodicFunctionLSTM([200, 200, 200], 128)
    model.load("./Trained/Additional_frequency_multistep_prediction/Network_LSTM_200_200_200_gaussian_noise_0,2/sample_1_k_128")

    # Test signals
    ## Regular signal
    # test_signal = PeriodicFunction(10, 0.016667)
    # test_signal.add_disturbing_signal(4, 0.4)

    ## Decaying high frequency component
    # test_signal = PeriodicFunction(10, 0.016667)
    # test_signal.add_disturbing_decaying_amp(0.01, 4, start, 0.4)

    ## Slow linear deviation
    # test_signal = PeriodicFunction(10, 0.016667)
    # test_signal.add_disturbing_signal(4, 0.4)
    # test_signal.add_linear_deviation(-0.01, start)

    ## Linear frequency deviation of low frequency component
    test_signal = PeriodicFunction(10, 0.016667)
    test_signal.add_disturbing_signal(4, 0.4)
    test_signal.add_frequency_deviation(0.0001, start)

    X, Y = build_dataset(test_signal, 128, 1, test=True)

    result = list(X[0])
    result.extend(list(Y))

    #plt.plot(result, color='b')

    np.expand_dims(np.array(result), axis=0)

    # Run the model
    for i in range(0, TESTSET_SIZE-1):
        yhat = model.predict(np.expand_dims(X[i-1], axis=0))

        # Add predictions to results table
        for j in range(127+i, TESTSET_SIZE+128):
            if j <= i+128+29:
                result[j] = np.append(result[j], yhat[0][j-i-129])
            else:
                result[j] = np.append(result[j], 0)

        #plt.plot(range(128+i, 158+i), yhat[0], color='r', alpha=0.1)


    scores = []
    
    # Compute absolute error
    for i in range(0, TESTSET_SIZE-1):
        predictions = []
        s = 1
        if i > 30:
            s += 1
        for j in range(s, len(result[128+i])):
            # Temp Fix
            if not result[128+i][j] == 0:
                predictions.append(result[128+i][j])
        scores.append(mean_absolute_difference(Y[i][0], predictions))



    """Write results to a csv file
    with open('./Anomaly_test/' + str(name) + '.csv', 'w', newline='') as outfile:
        w = csv.writer(outfile)
        w.writerow(titlerow)
        for row in result:
            w.writerow(row)"""

    """Write scores to a csv file
    with open('./Anomaly_test/' + str(name) + '.csv', 'w', newline='') as outfile:
        w = csv.writer(outfile)
        w.writerow(['Signal', 'MAE'])
        for i in range(0, len(scores)):
            w.writerow([Y[i][0], scores[i]])"""

    """# Plot predictions
    blue_patch = mpatches.Patch(color='b', label='Real signal')
    red_patch = mpatches.Patch(color='r', label='Overlayed predictions')
    plt.legend(handles=[blue_patch, red_patch])
    plt.tight_layout()
    plt.show()
    #plt.savefig("./Anomaly_test/Plot_" + str(name), dpi=1000)

    # Plot scores
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(Y, color='r')
    ax[1].plot(scores, color='r')
    ax[0].set_ylabel("Signal")
    ax[0].set_xlim(0, TESTSET_SIZE)
    ax[0].set_title('Anomalous signal')
    ax[1].set_ylabel("Votes")
    ax[1].set_xlim(0, TESTSET_SIZE)
    ax[1].set_title('Voting')
    plt.tight_layout()
    plt.savefig("./Anomaly_test/Plot_" + str(name), dpi=1000)
    """

    # Test different thresholds and determine at which steps anomalies are detected
    thresholds = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]
    false_positives = []
    delays = []
    start = 100

    # For each threshold -> get detected anomalies
    for threshold in thresholds:
        fp = 0
        delay = 0
        first = True
        for i in range(0, len(scores)):
            # Anomaly detected?
            if scores[i] > threshold:
                # False detection?
                if i+1 < start:
                    fp += 1
                elif first:
                    delay = i - start + 1
                    first = False
        if first:
            delays.append(None)
        else:
            delays.append(delay)
        false_positives.append(fp/start)

    plt.plot(thresholds, false_positives, color='r')
    plt.xlabel("Threshold")
    plt.ylabel("False positive rate")
    plt.savefig("./Anomaly_test/Plot_thresh-fpr_" + str(name), dpi=500)
    plt.clf()

    plt.plot(false_positives, delays, color='r')
    plt.xlim(0, 1)
    plt.xlabel("False positive rate")
    plt.ylabel("Delay")
    plt.savefig("./Anomaly_test/Plot_fpr-delay_" + str(name), dpi=500)

if __name__ == '__main__':
    seed(SEED)

    test_anomaly_detection()

    #train_and_test()