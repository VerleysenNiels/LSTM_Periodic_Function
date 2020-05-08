#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 19:54:30 2019

@author: Niels Verleysen

Class that represents a periodic function (sinusoid) with the possibility to add different types of noise.
First initialize the function and specify the amplitude and frequency of the sinusoid.
Then you can add noise to the function by calling the corresponding functions.
Get a value for a given timepoint by calling value.

An example is given below.
"""

import math
import random

import numpy as np

class PeriodicFunction:
    
    def __init__(self, amplitude, frequency):
        self.gaussian = False
        self.additionalF = False
        self.random_outliers = False
        self.increasing_amp = False
        self.increasing_freq = False
        self.increasing_both = False
        self.decaying_amp = False
        self.linear_deviation = False
        self.frequency_deviation = False
        self.amplitude = amplitude
        self.frequency = frequency * 6.283185
        
    def add_gaussian_noise(self, deviation):
        self.gaussian = True
        self.gaussian_dev = deviation
        
    def add_disturbing_signal(self, amplitude, frequency):
        self.additionalF = True
        self.additional_ampl = amplitude
        self.additional_freq = frequency * 6.283185

    # Amplitude of disturbing signal decays over time from start until it is zero
    def add_disturbing_decaying_amp(self, decay, amplitude, start, freq):
        self.decaying_amp = True
        self.decaying_amp_decay = decay
        self.decaying_amp_start = start
        self.decaying_amp_amplitude = amplitude
        self.decaying_amp_f = freq * 6.283185

    #Amplitude of disturbing signal increases linearly
    def add_disturbing_increasing_amp(self, a, b, freq):
        self.increasing_amp = True
        self.increasing_amp_a = a
        self.increasing_amp_b = b
        self.increasing_amp_f = freq * 6.283185

    # Frequency of disturbing signal increases linearly
    def add_disturbing_increasing_freq(self, a, b, ampl):
        self.increasing_freq = True
        self.increasing_freq_a = a
        self.increasing_freq_b = b
        self.increasing_freq_ampl = ampl

    # Frequency and amplitude of disturbing signal increases linearly
    def add_disturbing_increasing_both(self, a_freq, b_freq, a_amp, b_amp):
        self.increasing_both = True
        self.increasing_a_freq = a_freq
        self.increasing_b_freq = b_freq
        self.increasing_a_amp = a_amp
        self.increasing_b_amp = b_amp

    # Use poisson distribution with given lambda and multiplier
    def add_random_outliers(self, probability, lam, multi):
        self.random_outliers = True
        self.ro_probability = probability
        self.lam = lam
        self.multi = multi

    # Add a linear deviation to the signal from a given startpoint
    def add_linear_deviation(self, slope, start):
        self.linear_deviation = True
        self.linear_deviation_slope = slope
        self.linear_deviation_start = start

    # Add linear deviation of base frequency
    def add_frequency_deviation(self, slope, start):
        self.frequency_deviation = True
        self.frequency_deviation_slope = slope
        self.frequency_deviation_start = start
        
    def value(self, time):
        f = self.frequency
        if self.frequency_deviation:
            f = self.frequency_deviation_slope * max(0, time - self.frequency_deviation_start) + self.frequency

        val = self.amplitude * math.sin(time * f)
        
        if self.gaussian:
            noise = np.random.normal(0, self.gaussian_dev, 1)
            val = val + noise[0]
        
        if self.additionalF:
            noise = self.additional_ampl * math.sin(time * self.additional_freq)
            val = val + noise

        if self.increasing_freq:
            noise = self.increasing_freq_ampl * math.sin(time * (self.increasing_freq_a * time + self.increasing_freq_b) * 6.283185)
            val = val + noise

        if self.increasing_amp:
            noise = (self.increasing_amp_a * time + self.increasing_amp_b) * math.sin(time * self.increasing_amp_f)
            val = val + noise

        if self.increasing_both:
            noise = (self.increasing_a_amp * time + self.increasing_b_amp) * math.sin(time * (self.increasing_a_freq * time + self.increasing_b_freq) * 6.283185)
            val = val + noise

        if self.decaying_amp:
            if time < self.decaying_amp_start:
                # Decay has not started yet
                noise = self.decaying_amp_amplitude * math.sin(time * self.decaying_amp_f)
            else:
                # Decay has started, calculate amplitude of high frequency signal
                amplitude = self.decaying_amp_amplitude - (self.decaying_amp_decay * (time-self.decaying_amp_start))
                if amplitude < 0:
                    # High frequency signal is gone
                    noise = 0
                else:
                    # High frequency signal is decaying
                    noise = amplitude * math.sin(time * self.decaying_amp_f)
            val = val + noise

        if self.linear_deviation:
            noise = self.linear_deviation_slope * max(0, time - self.linear_deviation_start)
            val = val + noise

        if self.random_outliers:
            if self.ro_probability > random.uniform(0, 1):
                # Returned value is sampled from another distribution (here I use the Poisson distribution)
                val = np.random.poisson(self.lam, 1)[0] * self.multi

        return val
    
if __name__ == '__main__':
    function = PeriodicFunction(10, 0.016667)
    """
    v = function.value(15)
    print(v)
    function.add_gaussian_noise(1.3)
    v = function.value(15)
    print(v)
    function.add_disturbing_signal(2, 0.01)
    v = function.value(15)
    print(v)
    function.add_random_outliers(0.1, 4.0, 8)
    v = function.value(15)
    print(v)
    function.add_disturbing_increasing_amp(10, 1, 0.1)
    function.add_disturbing_increasing_freq(0.1, 1, 10)
    """

    #Test decay
    import matplotlib.pyplot as plt

    # Test signals
    ## Regular signal
    test_signal = PeriodicFunction(10, 0.016667)
    test_signal.add_disturbing_signal(4, 0.4)

    ## Decaying high frequency component
    #test_signal = PeriodicFunction(10, 0.016667)
    #test_signal.add_disturbing_decaying_amp(0.01, 4, 228, 0.4)

    ## Slow linear deviation
    #test_signal = PeriodicFunction(10, 0.016667)
    #test_signal.add_disturbing_signal(4, 0.4)
    #test_signal.add_linear_deviation(-0.01, 228)

    ## Linear frequency deviation of low frequency component
    #test_signal = PeriodicFunction(10, 0.016667)
    #test_signal.add_disturbing_signal(4, 0.4)
    #test_signal.add_frequency_deviation(0.0001, 228)

    vals = []
    for i in range(0, 828):
        vals.append(test_signal.value(i))

    plt.plot(vals)
    plt.ylabel("Signal value")
    plt.xlabel("Step")
    plt.xlim(0, 828)
    plt.ylim(-15, 15)
    plt.title('Regular signal')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.savefig("./NormalSignal", dpi=500)
