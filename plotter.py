#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:24:32 2020

@author: Niels Verleysen

This script can be used to plot the training history from the produced files by the main script.
These files are generated while training the network and are saved under Results/Training.

This script takes the history files or directories of history files and an export location as input.
The absolute error is then plotted and exported as an image.
The following example will generate the plots of the training progress on the Noiseless signal with amplitude 10 and save them in the plots folder.

python plotter.py ./Results/Training/Plots/ ./Results/Training/A_10_Gaussian_1/

"""
import sys
import os
import pathlib
import pickle
from matplotlib import pyplot

"""Checks the naming structure and extracts the sampling rate and k value from the name of the file"""
def get_file_info(file):
    name = file.split('_')
    if len(name) == 4 and name[0] == "sample" and name[2] == "k":
        return True, name[1], name[3]
    else:
        return False, None, None

"""Returns all files from given directory that follow the naming structure"""
def files_from_dir(dir):
    files = []
    for p in os.listdir(dir):
        path = os.path.join(dir, p)
        if os.path.isfile(path):
            check, sampling_rate, k = get_file_info(p)
            if check:
                files.append([path, sampling_rate, k])
        elif os.path.isdir(path):
            files.extend(files_from_dir(path))
    return files

"""Generates the plot of one training history file"""
def process_one(location, file, sampling_rate, k):
    pickle_in = open(file, "rb")
    history = pickle.load(pickle_in)
    pyplot.plot(history.history['mean_absolute_error'])
    pyplot.savefig(location + "/sample_" + str(sampling_rate) + "_k_" + str(k) + "_mae.png")
    pyplot.clf()

"""Main function runs process_one on every given file or file in a given folder after checking if the file has the right naming structure"""
if __name__ == '__main__':
    file_list = []
    for i in range(2, len(sys.argv)):
        path = pathlib.Path(sys.argv[i])
        if os.path.isdir(sys.argv[i]):
            file_list.extend(files_from_dir(path))
        elif os.path.isfile(path):
            check, sampling_rate, k = get_file_info(path)
            if check:
                file_list.append([path, sampling_rate, k])

    for file in file_list:
        process_one(sys.argv[1], file[0], file[1], file[2])
