#!/usr/bin/env python3
"""Expectation-Maximization implementation."""
import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform the expectation maximization for a GMM.
    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.
        verbose (bool): Whether to print information about the
        algorithm.
        Returns: pi, m, S, g, l, or (None, None, None, None, None) on failure.
            - pi is a numpy.ndarray of shape (k,) containing the priors for
            each cluster.
            - m is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster.
            - S is a numpy.ndarray of shape (k, d, d) containing the
            covariance matrices for each cluster.
            - g is a numpy.ndarray of shape (k, n) containing the
            posterior probabilities for each data point in each cluster.
            - l is the log likelihood of the model."""
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape

    lkhd_prev = 0
    pi, m, S = initialize(X, k)

    for i in range(iterations + 1):
        if i != 0:
            lkhd_prev = lkhd
            pi, m, S = maximization(X, g)
        g, lkhd = expectation(X, pi, m, S)
        if verbose:
            if i % 10 == 0 or i == iterations or abs(lkhd - lkhd_prev) <= tol:
                print("Log Likelihood after {} iterations: {}".
                      format(i, lkhd.round(5)))
        if abs(lkhd - lkhd_prev) <= tol:
            break

    return pi, m, S, g, lkhd
