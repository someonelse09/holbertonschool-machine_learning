#!/usr/bin/env python3
"""This module includes the function expectation
that calculates the expectation step in the EM algorithm for a GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Args:
        X is a numpy.ndarray of shape (n, d)
         containing the data set
        pi is a numpy.ndarray of shape (k,)
         containing the priors for each cluster
        m is a numpy.ndarray of shape (k, d) containing
         the centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing
         the covariance matrices for each cluster
        You may use at most 1 loop
    Returns:
        g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing
         the posterior probabilities for each data point in each cluster
        l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None
    if S.shape != (k, d, d):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    total_p = np.sum(g, axis=0)
    if np.any(total_p == 0):
        return None, None

    g /= total_p

    log_likelihood = np.sum(np.log(total_p))

    return g, log_likelihood
