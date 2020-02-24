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
import numpy as np

class PeriodicFunction:
    
    def __init__(self, amplitude, frequency):
        self.gaussian = False
        self.additionalF = False
        self.asymmetric = False
        self.amplitude = amplitude
        self.frequency = frequency * 6.283185
        
    def add_gaussian_noise(self, deviation):
        self.gaussian = True
        self.gaussian_dev = deviation
        
    def add_additional_frequency(self, amplitude, frequency):
        self.additionalF = True
        self.additional_ampl = amplitude
        self.additional_freq = frequency * 6.283185
        
    def add_asymmetric_distributed_noise(self):
        print("ToDo")
        
    def value(self, time):
        val = self.amplitude * math.sin(time * self.frequency)
        
        if self.gaussian:
            noise = np.random.normal(0, self.gaussian_dev, 1)
            val = val + noise[0]
        
        if self.additionalF:
            noise = self.additional_ampl * math.sin(time * self.additional_freq)
            val = val + noise
        
        return val
    
if __name__ == '__main__':
    function = PeriodicFunction(15, 0.3)
    v = function.value(15)
    print(v)
    function.add_gaussian_noise(1.3)
    v = function.value(15)
    print(v)
    function.add_additional_frequency(2, 0.01)
    v = function.value(15)
    print(v)
