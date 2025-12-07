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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    # Initialize centroids using multivariate uniform distribution
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    # Perform K-means iterations
    for i in range(iterations):
        # Store previous centroids to check for convergence
        C_old = C.copy()

        # Assignment step: assign each point to nearest centroid
        # Calculate distances from each point to each centroid
        # Shape: (n, k) - distance from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # Get index of closest centroid for each point
        clss = np.argmin(distances, axis=1)

        # Update step: recalculate centroids
        for j in range(k):
            # Get all points assigned to cluster j
            cluster_points = X[clss == j]

            if cluster_points.shape[0] == 0:
                # Cluster is empty, reinitialize its centroid
                C[j] = np.random.uniform(low=min_vals, high=max_vals, size=(d,))
            else:
                # Update centroid as mean of assigned points
                C[j] = np.mean(cluster_points, axis=0)

        # Check for convergence (no change in centroids)
        if np.allclose(C, C_old):
            break

    # Final assignment with updated centroids
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
