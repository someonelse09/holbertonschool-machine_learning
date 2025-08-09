#!/usr/bin/env python3
""" This module includes a function that 
takes the transpose of given matrix """


def matrix_transpose(matrix):
    """ this function returns the 
    transpose of the argument matrix """

    number_of_cols = len(matrix[0])
    new_matrix = []
    for i in range(number_of_cols):
        new_column = []
        for row in matrix:
            new_column.append(row[i])
        new_matrix.append(new_column)
    return new_matrix
