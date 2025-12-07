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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None

    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    # Check dimension consistency
    if m.shape[0] != d:
        return None

    if S.shape[0] != d or S.shape[1] != d:
        return None

    # Multivariate Gaussian PDF formula:
    # P(x) = (1 / sqrt((2π)^d * |Σ|)) * exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ))

    # Calculate determinant of covariance matrix
    det_S = np.linalg.det(S)

    # Check if determinant is valid (positive for valid covariance matrix)
    if det_S <= 0:
        return None

    # Calculate inverse of covariance matrix
    S_inv = np.linalg.inv(S)

    # Calculate normalization coefficient
    # coef = 1 / sqrt((2π)^d * |Σ|)
    coef = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_S)

    # Calculate (X - m) for all points
    # Shape: (n, d)
    X_centered = X - m

    # Calculate Mahalanobis distance for each point
    # For each point: (x-μ)^T * Σ^(-1) * (x-μ)
    # Using matrix multiplication: (X-m) @ S_inv @ (X-m).T
    # Shape: (n, d) @ (d, d) = (n, d)
    mahalanobis_part = np.sum(X_centered @ S_inv * X_centered, axis=1)

    # Calculate PDF values
    # P = coef * exp(-0.5 * mahalanobis_distance)
    P = coef * np.exp(-0.5 * mahalanobis_part)

    # Ensure minimum value of 1e-300
    P = np.maximum(P, 1e-300)

    return P
