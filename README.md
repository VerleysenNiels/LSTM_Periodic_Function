# LSTM_Periodic_Function
Experiment for my thesis. In this repository I will test the predictive quality of an LSTM on a periodic function with different sampling rates, types (and levels) of noise and amount of previous values given as input. The training progress will be saved as raw data under Results/Training. The trained weights are stored in the Weights folder. Finally the trained network is tested on the periodic function and the result is stored in a table in the Results folder.

The network model, the training signal and the other parameters can be changed in the main.py script.
The training progress can be plotted with the plotter.py script, which expects a location to save these plots and one or multiple paths to training progress files or directories containing these files. This script will then collect all these files and generate a plot of the mean absolute error of each.
