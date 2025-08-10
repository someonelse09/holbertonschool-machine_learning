#!/usr/bin/env python3
""" This module includes function to
slice the matrix using numpy """



def np_slice(matrix, axes={}):
    """ This function slices the matrix
    based on the given axis """

    slices = []

    for axis in range(matrix.ndim):
        if axis in axes:
            slice_tuple = axes[axis]
            slices.append(slice(*slice_tuple))
        else:
            slices.append(slice(None))

    return matrix[tuple(slices)]
