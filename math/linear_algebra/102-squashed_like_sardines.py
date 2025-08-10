#!/usr/bin/env python3
""" This module contains function
that is kind of the generalisation of 
the previosly written cat_matrices2D """


matrix_shape = __import__('2-size_me_please').matrix_shape


def cat_matrices(mat1, mat2, axis=0):
    """ This function concats given
    two matrices of any dimension """

    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 is None or shape2 is None:
        return None
    if axis >= len(shape1) or axis >= len(shape2)):
        return None
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis or shape1[i] != shape2[i]:
            return None
    if axis == 0:
        return mat1 + mat2
    result = []
    for i in range(len(mat1)):
        concatenated = cat_matrices(mat1, mat2, axis - 1)
        if concatenated is None:
            return None
        result.append(concatenated)
    return result
