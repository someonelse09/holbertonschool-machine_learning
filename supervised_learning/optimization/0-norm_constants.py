#!/usr/bin/env python3
"""This module includes the
function to normalize constants"""

import numpy as np


def normalization_constants(X):
    """"This function returns mean and
    standard deviation of each feature, respectively"""
    mean = np.mean(X, axis=0)
    standard_deviation = np.std(X, axis=0)
    return mean, standard_deviation
