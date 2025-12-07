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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None

    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    # Check that priors sum to 1
    if not np.isclose(np.sum(pi), 1):
        return None, None
    # Calculate likelihood for each cluster
    # Shape: (k, n) - likelihood of each point under each cluster's Gaussian
    likelihoods = np.zeros((k, n))

    # Looping through each cluster to calculate PDF
    for i in range(k):
        """Calculating the PDF of all points under cluster i's Gaussian"""
        pdf_values = pdf(X, m[i], S[i])
        if pdf_values is None:
            return None, None
        # Multiply by prior for this cluster
        # Shape: (n,)
        likelihoods[i] = pi[i] * pdf_values

    # Calculate marginal probability (total likelihood for each point)
    # Sum across all clusters for each point
    # Shape: (n,)
    marginal = np.sum(likelihoods, axis=0)
    if np.any(marginal == 0):
        return None, None

    # Calculate posterior probabilities (responsibilities)
    # g[i, j] = P(cluster i | data point j)
    # Using Bayes' theorem: P(cluster|point) = P(point|cluster) * P(cluster) / P(point)
    # Shape: (k, n)
    g = likelihoods / marginal
    # Calculate total log likelihood
    # Sum of log of marginal probabilities
    l = np.sum(np.log(marginal))

    return g, l
