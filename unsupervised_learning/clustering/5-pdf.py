#!/usr/bin/env python3
"""This module contains the function that calculates
the probability density function of a Gaussian distribution"""

import numpy as np

pi = 3.141592653


def pdf(X, m, S):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing
         the data points whose PDF should be evaluated
        m is a numpy.ndarray of shape (d,) containing
         the mean of the distribution
        S is a numpy.ndarray of shape (d, d) containing
         the covariance of the distribution
        You are not allowed to use any loops
        You are not allowed to use the function
         numpy.diag or the method numpy.ndarray.diagonal
    Returns:
        P, or None on failure
        P is a numpy.ndarray of shape (n,)
         containing the PDF values for each data point
        All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d:
        return None
    if S.shape != (d, d):
        return None

    try:
        det_S = np.linalg.det(S)
        if det_S <= 0:
            return None

        inv_S = np.linalg.inv(S)
    except (np.linalg.LinAlgError, ValueError):
        return None

    diff = X - m

    quad = np.sum((diff @ inv_S) * diff, axis=1)

    norm_const = np.sqrt(((2 * np.pi) ** d) * det_S)

    P = np.exp(-0.5 * quad) / norm_const

    P = np.maximum(P, 1e-300)

    return P
