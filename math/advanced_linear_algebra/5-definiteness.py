#!/usr/bin/env python3
"""This module includes the function
that calculates the definiteness of a matrix"""

import numpy as np


def definiteness(matrix):
    """
    Args:
        matrix is a numpy.ndarray of shape (n, n)
         whose definiteness should be calculated
    If matrix is not a numpy.ndarray, raise a TypeError
     with the message matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None
    Returns:
        the string Positive definite, Positive semi-definite,
         Negative semi-definite, Negative definite, or Indefinite
          if the matrix is positive definite, positive semi-definite,
           negative semi-definite, negative definite of indefinite,
            respectively
    If matrix does not fit any of the above categories, return None
    You may import numpy as np
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if matrix.ndim != 2:
        return None

    n, m = matrix.shape
    if n != m:
        return None

    # Must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Eigenvalues (real, ordered) for symmetric matrices
    eigenvalues = np.linalg.eigvalsh(matrix)

    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)

    # Classification logic
    if min_eig > 0:
        return "Positive definite"
    if min_eig >= 0 and max_eig >= 0:
        return "Positive semi-definite"
    if max_eig < 0:
        return "Negative definite"
    if max_eig <= 0 and min_eig <= 0:
        return "Negative semi-definite"

    return "Indefinite"
