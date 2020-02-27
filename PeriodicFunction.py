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
        self.amplitude = amplitude
        self.frequency = frequency * 6.283185
        
    def add_gaussian_noise(self, deviation):
        self.gaussian = True
        self.gaussian_dev = deviation
        
    def add_disturbing_frequency(self, amplitude, frequency):
        self.additionalF = True
        self.additional_ampl = amplitude
        self.additional_freq = frequency * 6.283185

    #Amplitude of disturbing frequency increases linearly
    def add_disturbingf_increasing_amp(self, a, b, freq):
        self.increasing_amp = True
        self.increasing_amp_a = a
        self.increasing_amp_b = b
        self.increasing_amp_f = freq * 6.283185

    # Frequency of disturbing frequency increases linearly
    def add_disturbingf_increasing_freq(self, a, b, ampl):
        self.increasing_freq = True
        self.increasing_freq_a = a
        self.increasing_freq_b = b
        self.increasing_freq_ampl = ampl

    # Use poisson distribution with given lambda and multiplier
    def add_random_outliers(self, probability, lam, multi):
        self.random_outliers = True
        self.ro_probability = probability
        self.lam = lam
        self.multi = multi
        
    def value(self, time):
        val = self.amplitude * math.sin(time * self.frequency)
        
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

        if self.random_outliers:
            if self.ro_probability > random.uniform(0, 1):
                # Returned value is sampled from another distribution (here I use the Poisson distribution)
                val = np.random.poisson(self.lam, 1)[0] * self.multi

        return val
    
if __name__ == '__main__':
    function = PeriodicFunction(15, 0.3)
    v = function.value(15)
    print(v)
    function.add_gaussian_noise(1.3)
    v = function.value(15)
    print(v)
    function.add_disturbing_frequency(2, 0.01)
    v = function.value(15)
    print(v)
    function.add_random_outliers(0.1, 4.0, 8)
    v = function.value(15)
    print(v)
    function.add_disturbingf_increasing_amp(10, 1, 0.1)
    function.add_disturbingf_increasing_freq(0.1, 1, 10)
