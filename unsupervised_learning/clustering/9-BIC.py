#!/usr/bin/env python3
"""
This module contains the function BIC that finds the best
 number of clusters for a GMM using the Bayesian Information Criterion
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        kmin is a positive integer containing
         the minimum number of clusters to check for (inclusive)
        kmax is a positive integer containing
         the maximum number of clusters to check for (inclusive)
        If kmax is None, kmax should be set
         to the maximum number of clusters possible
        iterations is a positive integer containing
         the maximum number of iterations for the EM algorithm
        tol is a non-negative float containing
         the tolerance for the EM algorithm
        verbose is a boolean that determines if the EM algorithm
         should print information to the standard output\
        You may use at most 1 loop
    Returns:
        best_k, best_result, l, b, or None, None, None, None on failure
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
        pi is a numpy.ndarray of shape (k,) containing
         the cluster priors for the best number of clusters
        m is a numpy.ndarray of shape (k, d) containing
         the centroid means for the best number of clusters
        S is a numpy.ndarray of shape (k, d, d) containing
         the covariance matrices for the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing
         the log likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing
         the BIC value for each cluster size tested
        Use: BIC = p * ln(n) - 2 * l
        p is the number of parameters required for the model
        n is the number of data points used to create the model
        l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    # Set kmax to maximum possible if not provided
    if kmax is None:
        kmax = n

    # Validate kmin and kmax
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None

    if kmin > kmax:
        return None, None, None, None

    if kmax > n:
        return None, None, None, None

    # Validate other parameters
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    # Import EM algorithm
    expectation_maximization = __import__('8-EM').expectation_maximization

    # Initialize arrays to store results
    num_tests = kmax - kmin + 1
    l = np.zeros(num_tests)
    b = np.zeros(num_tests)
    results = []

    # Test each value of k
    for idx, k in enumerate(range(kmin, kmax + 1)):
        # Run EM algorithm for current k
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        # Store results
        results.append((pi, m, S))
        l[idx] = log_likelihood

        # Calculate number of parameters for the model
        # p = k * d (means) + k * d * (d + 1) / 2 (covariances) + (k - 1) (priors)
        #
        # Breaking it down:
        # - Means: k clusters * d dimensions = k * d parameters
        # - Covariances: k clusters * d*(d+1)/2 unique values (symmetric matrix)
        # - Priors: k - 1 (since they sum to 1, last one is determined)

        # Means parameters
        p_means = k * d

        # Covariance parameters (upper triangle of symmetric matrix)
        p_cov = k * d * (d + 1) // 2

        # Prior parameters (k - 1 since they sum to 1)
        p_priors = k - 1

        # Total parameters
        p = p_means + p_cov + p_priors

        # Calculate BIC: BIC = p * ln(n) - 2 * l
        b[idx] = p * np.log(n) - 2 * log_likelihood

    # Find the best k (minimum BIC)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
