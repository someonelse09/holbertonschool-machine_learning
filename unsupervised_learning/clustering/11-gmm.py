#!/usr/bin/env python3
"""GMM with sklearn implementation."""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset using sklearn.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters

    Returns:
        tuple: (pi, m, S, clss, bic)
            - pi is numpy.ndarray of shape (k,) containing the priors
            - m is numpy.ndarray of shape (k, d) containing the means
            - S is numpy.ndarray of shape (k, d, d) containing the covariances
            - clss is numpy.ndarray of shape (n,) containing cluster labels
            - bic is numpy.ndarray of shape (1,) containing the BIC value
    """
    if not hasattr(X, "shape") or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
