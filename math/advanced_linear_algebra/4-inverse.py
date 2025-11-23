#!/usr/bin/env python3
"""This module includes the function
that calculates the inverse of a matrix"""


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


def minor(matrix):
    """
    Args:
        matrix is a list of lists whose minor matrix should be calculated
        If matrix is not a list of lists, raise a TypeError
         with the message matrix must be a list of lists
        If matrix is not square or is empty, raise a ValueError
         with the message matrix must be a non-empty square matrix
    Returns:
        the minor matrix of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    size = len(matrix)
    if size == 1:
        return [[1]]

    minor_matrix = []
    for i in range(len(matrix)):
        row_i_of_minor = []
        for j in range(len(matrix[0])):
            sub = [row[:j] + row[j+1:]
                   for k, row in enumerate(matrix)
                   if k != i]
            row_i_of_minor.append(determinant(sub))
        minor_matrix.append(row_i_of_minor)
    return minor_matrix


def cofactor(matrix):
    """
    Args:
        matrix is a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError
     with the message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError
     with the message matrix must be a non-empty square matrix
    Returns:
        the cofactor matrix of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    cofactor = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix[0])):
            cofactor_row.append(((-1) ** (i+j)) * minor_matrix[i][j])
        cofactor.append(cofactor_row)
    return cofactor


def adjugate(matrix):
    """
    Args:
        matrix is a list of lists whose adjugate matrix should be calculated
    If matrix is not a list of lists, raise a TypeError
     with the message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError
     with the message matrix must be a non-empty square matrix
    Returns: the adjugate matrix of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    cofact = cofactor(matrix)
    adj = [[row[j] for row in cofact] for j in range(len(cofact[0]))]

    return adj


def inverse(matrix):
    """
    Args:
        matrix is a list of lists whose inverse should be calculated
    If matrix is not a list of lists, raise a TypeError
     with the message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError
     with the message matrix must be a non-empty square matrix
    Returns:
        the inverse of matrix, or None if matrix is singular
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Check for non-square rows
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    
    deter = determinant(matrix)
    if deter == 0:
        return None
    
    adjugate_matrix = adjugate(matrix)
    inverse_matrix = []
    for i in range(len(matrix)):
        inverted_row = []
        for j in range(len(matrix[0])):
            inverted_row.append((1 / deter) * adjugate_matrix[i][j])
        inverse_matrix.append(inverted_row)
    return inverse_matrix
