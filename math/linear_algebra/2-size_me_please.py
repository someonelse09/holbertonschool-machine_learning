#!/usr/bin/env python3


def matrix_shape(matrix):
    """Return the shape of a matrix as a tuple"""
    if not isinstance(matrix, list):
        return []
    shape = []
    current = matrix
    while isinstance(current, list) and len(current) > 0:
        shape.append(len(current))
        current = current[0]

    return shape
