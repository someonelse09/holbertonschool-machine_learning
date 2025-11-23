#!/usr/bin/env python3
"""This module includes the function
that calculates the determinant of a matrix"""


def determinant(matrix):
    """
    Args:
        matrix is a list of lists whose determinant should be calculated
        If matrix is not a list of lists, raise a TypeError
         with the message matrix must be a list of lists
        If matrix is not square, raise a ValueError
         with the message matrix must be a square matrix
        The list [[]] represents a 0x0 matrix
    Returns:
        the determinant of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i in range(len(matrix)):
        # Create minor matrix by removing row 0 and column i
        minor = [row[:i] + row[i + 1:] for row in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * determinant(minor)

    return det
