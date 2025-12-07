#!/usr/bin/env python3
"""
K-means clustering algorithm module
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           n is the number of data points
           d is the number of dimensions for each data point
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means
        clss: numpy.ndarray of shape (n,) containing the cluster index for each point
        Returns None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    C = np.random.uniform(min_vals, max_vals, (k, d))

    for i in range(iterations):
        C_old = C.copy()

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        clss = np.argmin(distances, axis=1)

        for j in range(k):
            cluster_points = X[clss == j]

            if len(cluster_points) == 0:
                C[j] = np.random.uniform(min_vals, max_vals, (d,))
            else:
                C[j] = np.mean(cluster_points, axis=0)

        if np.allclose(C, C_old):
            break

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
