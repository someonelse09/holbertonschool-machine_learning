#!/usr/bin/env python3
""" This module contains the function add_matrices """


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """ This function two matrices element-wise """

    res_matrix = []
    if (matrix_shape(mat1) != matrix_shape(mat2)):
        return None

    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat1[0])):
            new_row.append(mat1[i][j] + mat2[i][j])
        res_matrix.append(new_row)
    return res_matrix
