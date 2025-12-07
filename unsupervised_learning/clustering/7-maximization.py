#!/usr/bin/env python3
"""This module includes the function maximization
that calculates the maximization step in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        g is a numpy.ndarray of shape (k, n) containing the
         posterior probabilities for each data point in each cluster
        You may use at most 1 loop
    Returns:
        pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing
         the updated priors for each cluster
        m is a numpy.ndarray of shape (k, d) containing the
         updated centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing
         the updated covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    # Check that responsibilities sum to 1 for each point
    sums = np.sum(g, axis=0)
    if not np.allclose(sums, 1):
        return None, None, None

    # Calculate effective number of points assigned to each cluster
    # N[i] = sum of responsibilities for cluster i across all points
    # Shape: (k,)
    N = np.sum(g, axis=1)

    # Check for empty clusters (no points assigned)
    if np.any(N == 0):
        return None, None, None

    # Update priors
    # π[i] = N[i] / n (proportion of points assigned to cluster i)
    # Shape: (k,)
    pi = N / n

    # Update means
    # μ[i] = (1/N[i]) * sum of (g[i,j] * x[j]) for all points j
    # Shape: (k, d)
    # g.T @ X gives sum of weighted points, divide by N to get mean
    m = (g @ X) / N[:, np.newaxis]

    # Update covariance matrices
    # Σ[i] = (1/N[i]) * sum of g[i,j] * (x[j] - μ[i]) * (x[j] - μ[i])^T
    # Shape: (k, d, d)
    S = np.zeros((k, d, d))

    # Loop through each cluster to calculate covariance
    for i in range(k):
        # Calculate difference from mean for all points
        # Shape: (n, d)
        X_centered = X - m[i]

        # Weight each difference by the responsibility
        # g[i] has shape (n,), need to make it (n, 1) for broadcasting
        # Shape: (n, d)
        weighted_diff = X_centered * g[i, :, np.newaxis]

        # Calculate covariance matrix
        # (n, d).T @ (n, d) = (d, d)
        S[i] = (weighted_diff.T @ X_centered) / N[i]

    return pi, m, S
