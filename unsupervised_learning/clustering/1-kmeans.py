#!/usr/bin/env python3
"""This module includes the function
that performs K-means on a dataset"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing the
         maximum number of iterations that should be performed
        If no change in the cluster centroids occurs between
         iterations, your function should return
        Initialize the cluster centroids using a multivariate
         uniform distribution (based on0-initialize.py)
        If a cluster contains no data points during the update step,
         reinitialize its centroid
        You should use numpy.random.uniform exactly twice
        You may use at most 2 loops
    Returns:
        C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing
         the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,)
         containing the index of the cluster in C
          that each data point belongs to
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
    c = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for i in range(iterations):
        c_old = c.copy()

        # Assignment step: assign each point to nearest centroid
        # Calculate distances from each point to each centroid
        # Shape: (n, k) - distance from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - c, axis=2)

        # Get index of closest centroid for each point
        cls = np.argmin(distances, axis=1)
        # Update step: recalculate centroids
        for j in range(k):
            # Get all points assigned to cluster j
            cluster_points = X[cls == j]
            if cluster_points.shape[0] == 0:
                # Cluster is empty, reinitialize its centroid
                c[j] = np.random.uniform(low=min_vals, high=max_vals, size=(d,))
            else:
                # Update centroid as mean of assigned points
                c[j] = np.mean(cluster_points, axis=0)
        if np.allclose(c, c_old):
            break
    # Final assignment with updated centroids
    distances = np.linalg.norm(X[:, np.newaxis] - c, axis=2)
    cls = np.argmin(distances, axis=1)

    return c, cls
