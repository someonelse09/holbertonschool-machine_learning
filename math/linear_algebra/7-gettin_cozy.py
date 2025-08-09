#!/usr/bin/env python3
""" This module includes cat_matrices2D """


def cat_matrices2D(mat1, mat2, axis=0):
    """This function concatenates
    mat1 to mat2 based on the axis """

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_matrix = []
        for row1 in mat1:
            new_matrix.append(row1[:])
        for row2 in mat2:
            new_matrix.append(row2[:])
        return new_matrix

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_matrix = []
        for i in range(len(mat1)):
            new_row = mat1[i][:] + mat2[i][:]
            new_matrix.append(new_row)
        return new_matrix
    else:
        return None
