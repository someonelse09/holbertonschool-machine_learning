#!/usr/bin/env python3
""" This module includes function np_cat """


def np_cat(mat1, mat2, axis=0):
    """ This function concatenates the two matrices """

    return __import__('numpy').concatenate((mat2, mat2), axis=axis)
