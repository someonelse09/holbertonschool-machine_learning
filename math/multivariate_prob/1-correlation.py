#!/usr/bin/env python3
"""This module includes the function
 that calculates a correlation matrix"""

import numpy as np


def correlation(C):
    """
    Args:
        C is a numpy.ndarray of shape (d, d)
         containing a covariance matrix
        d is the number of dimensions
        If C is not a numpy.ndarray, raise a TypeError
         with the message C must be a numpy.ndarray
        If C does not have shape (d, d), raise a ValueError
         with the message C must be a 2D square matrix
    Returns:
        a numpy.ndarray of shape (d, d)
         containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

        # Check if C is 2D and square
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

        # Get the standard deviations (square root of diagonal elements)
    std_devs = np.sqrt(np.diag(C))

    # Create the outer product of standard deviations
    # This gives us a matrix where element (i,j) = std_i * std_j
    std_outer = np.outer(std_devs, std_devs)

    # Correlation = Covariance / (std_i * std_j)
    correlation_matrix = C / std_outer

    return correlation_matrix
