#!/usr/bin/env python3
"""This module includes the function pca that performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
        var is the fraction of the variance that
        the PCA transformation should maintain
    Returns:
        the weights matrix, W, that maintains
        var fraction of X's original variance
        W is a numpy.ndarray of shape (d, nd) where
        nd is the new dimensionality of the transformed X
    """

    # Left Singular vectors, Singular Values and Right Singular Vectors
    u, s, v = np.linalg.svd(X)
    ratios = list(x / np.sum(s) for x in s)
    variance = np.cumsum(ratios)
    nd = np.argwhere(variance >= var)[0, 0] + 1
    w = v.T[:, :nd]
    return w

