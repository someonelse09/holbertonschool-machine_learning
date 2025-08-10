#!/usr/bin/env python3
""" This module includes a function
to add two multi-dimensional arrays """


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices(mat1, mat2):
    """ This function allows its users to
    add two matrices(multi-dimensional case being considered) """

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    if not isinstance(mat1, list) and not isinstance(mat2, list):
        return mat1 + mat2

    if isinstance(mat1, list) != isinstance(mat2, list) or len(mat1) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        element_sum = add_matrices(mat1[i], mat2[i])
        if element_sum is None:
            return None
        result.append(element_sum)
    return result
