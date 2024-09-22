# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:19:49 2021

@author: hieu1
"""

import numpy as np

class DPCM():
    def __init__(self, diff_table):
        self.diff_table = np.array(diff_table)
    def encode(self, wave):
        wave = np.array(wave)
        if len(wave.shape)==2:
            return np.vstack([self.encode(wave[:,0]),self.encode(wave[:,1])]).T
        symbols = np.zeros(len(wave), dtype=np.uint)
        prediction = 0
        for i, model in enumerate(wave):
            predictions = prediction + self.diff_table
            abs_error = np.abs(predictions - model)
            diff_index = np.argmin(abs_error)
            symbols[i] = diff_index
            prediction += self.diff_table[diff_index]
        return symbols
    def decode(self, symbols):
        if len(symbols.shape)==2:
            return np.vstack([self.decode(symbols[:,0]),self.decode(symbols[:,1])]).T
        wave = np.zeros(len(symbols), dtype=np.double)
        prediction = 0
        for i, diff_index in enumerate(symbols):
            prediction += self.diff_table[diff_index]
            wave[i] = prediction
        return wave