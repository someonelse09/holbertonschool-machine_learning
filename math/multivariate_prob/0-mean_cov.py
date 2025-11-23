#!/usr/bin/env python3
"""This module includes the function
 that calculates the mean and covariance of a data set"""

import numpy as np

def mean_cov(X):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
    If X is not a 2D numpy.ndarray, raise a TypeError
     with the message X must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError
     with the message X must contain multiple data points
    Returns:
        mean, cov:
    mean is a numpy.ndarray of shape
     (1, d) containing the mean of the data set
    cov is a numpy.ndarray of shape
     (d, d) containing the covariance matrix of the data set
    You are not allowed to use the function numpy.cov
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")

    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    X_centered = X - mean

    # Covariance formula: (1/(n-1)) * X_centered.T @ X_centered
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
