#!/usr/bin/env python3
""" This module contains the function called mat_mul """


def mat_mul(mat1, mat2):
    """ This function returns the
    product of matrices mat1 and mat2 """

    if len(mat1[0]) != len(mat2):
        return None

    res_matrix = []
    row_index = 0
    while (row_index < len(mat1)):
        new_row = []
        for i in range(len(mat2[0])):
            dot_P = 0
            for j in range(len(mat2)):
                dot_P += mat1[row_index][j]*mat2[j][i]
            new_row.append(dot_P)
        res_matrix.append(new_row)
        row_index += 1
    return res_matrix
