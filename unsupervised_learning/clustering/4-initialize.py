#!/usr/bin/env python3
"""This module includes the function initialize
that initializes variables for a Gaussian Mixture Model"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        k is a positive integer containing the number of clusters
        You are not allowed to use any loops
    Returns:
        pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the
         priors for each cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid
         means for each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
         matrices for each cluster, initialized as identity matrices
        You should use kmeans = __import__('1-kmeans').kmeans
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    if k > n:
        return None, None, None
    # Initialize priors evenly (each cluster has equal probability)
    # Shape: (k,)
    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None
    s = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, s
