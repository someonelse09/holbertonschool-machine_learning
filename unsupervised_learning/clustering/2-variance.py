#!/usr/bin/env python3
"""This module includes the function variance that
calculates the total intra-cluster variance for a data set"""

import numpy as np


def variance(X, C):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing
         the centroid means for each cluster
    Returns:
        var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    n, d = X.shape
    k, d_c = C.shape

    if d != d_c:
        return None
    if k == 0:
        return None

    # Calculate distances from each point to each centroid
    # X[:, np.newaxis] shape: (n, 1, d)
    # C shape: (k, d)
    # After broadcasting: (n, k, d)
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    min_distances = np.min(distances, axis=1)
    variance = np.sum(min_distances ** 2)

    return variance
