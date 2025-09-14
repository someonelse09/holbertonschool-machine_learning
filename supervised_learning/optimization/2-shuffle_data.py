#!/usr/bin/env python3
"""This module includes the
function called shuffle_data"""

import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the
    data points in two matrices the same way"""
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
