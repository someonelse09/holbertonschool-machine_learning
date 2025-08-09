#!/usr/bin/env python3
""" This module includes shape function """


matrix_shape = __import__('2-size_me_please').matrix_shape


def np_shape(matrix):
    """ This function calculates the shape of the matrix """

    return tuple(matrix_shape(matrix))
