#!/usr/bin/env python3
""" This module includes function np_cat """


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ This function concatenates the two matrices """

    return np.concatenate((mat1, mat2), axis=axis)
