#!/usr/bin/env python3
"""Gaussian Mixture Model implementation."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Optimized version of BIC calculation"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    k_range = kmax - kmin + 1
    L = np.zeros(k_range)
    b = np.zeros(k_range)
    best_k = None
    best_result = None
    best_BIC = np.inf

    for i in range(kmin, kmax + 1):
        idx = i - kmin
        pi, m, S, g, li = expectation_maximization(
            X, i, iterations, tol, verbose)

        p = (i * d) + (i * d * (d + 1) // 2) + (i - 1)
        current_BIC = p * np.log(n) - 2 * li

        L[idx] = li
        b[idx] = current_BIC

        if current_BIC < best_BIC:
            best_k = i
            best_result = (pi, m, S)
            best_BIC = current_BIC

    return best_k, best_result, L, b
