#!/usr/bin/env python3
"""This module includes the function expectation_maximization
that performs the expectation maximization for a GMM"""

import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing
         the maximum number of iterations for the algorithm
        tol is a non-negative float containing tolerance of the
         log likelihood, used to determine early stopping i.e. if
         the difference is less than or equal to tol you should stop the algorithm
        verbose is a boolean that determines if
         you should print information about the algorithm
        If True, print Log Likelihood after {i} iterations:
         {l} every 10 iterations and after the last iteration
        {i} is the number of iterations of the EM algorithm
        {l} is the log likelihood, rounded to 5 decimal places
        You should use:
        initialize = __import__('4-initialize').initialize
        expectation = __import__('6-expectation').expectation
        maximization = __import__('7-maximization').maximization
        You may use at most 1 loop
    Returns:
        pi, m, S, g, l, or None, None, None, None, None on failure
        pi is a numpy.ndarray of shape (k,)
         containing the priors for each cluster
        m is a numpy.ndarray of shape (k, d)
         containing the centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d)
         containing the covariance matrices for each cluster
        g is a numpy.ndarray of shape (k, n) containing
         the probabilities for each data point in each cluster
        l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape

    if k > n:
        return None, None, None, None, None

    # Import required functions
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    # Initialize parameters
    pi, m, S = initialize(X, k)

    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Initialize previous log likelihood for convergence check
    l_prev = 0

    # Initialize previous log likelihood for convergence check
    l_prev = None

    # EM algorithm loop
    for i in range(iterations):
        # E-step: Calculate posterior probabilities
        g, l = expectation(X, pi, m, S)

        if g is None or l is None:
            return None, None, None, None, None

        # Print log likelihood if verbose
        if verbose and (i % 10 == 0):
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

        # Check for convergence (skip first iteration)
        if l_prev is not None and abs(l - l_prev) <= tol:
            # Print final iteration if verbose
            if verbose:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")
            break

        # Store current log likelihood for next iteration
        l_prev = l

        # M-step: Update parameters
        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None
    else:
        # Loop completed without early stopping
        # Need to run one final E-step to get final g and l
        g, l = expectation(X, pi, m, S)

        if g is None or l is None:
            return None, None, None, None, None

        # Print final log likelihood if verbose
        if verbose:
            print(f"Log Likelihood after {iterations} iterations: {l:.5f}")

    return pi, m, S, g, l

