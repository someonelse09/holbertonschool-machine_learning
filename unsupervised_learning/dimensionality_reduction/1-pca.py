#!/usr/bin/env python3
"""This module includes the function pca
that performs PCA on a dataset"""

import numpy as np


def pca(X, ndim):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        ndim is the new dimensionality of the transformed X
    Returns:
        T, a numpy.ndarray of shape (n, ndim)
         containing the transformed version of X
    """
    mean = np.mean(X, axis=0)
    a = X - mean
    u, s, v = np.linalg.svd(a)
    w = v.T[:, :ndim]
    t = np.matmul(a, w)
    return t
