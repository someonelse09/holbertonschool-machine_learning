#!/usr/bin/env python3
"""Perform K-means with sklearn."""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters

    Returns:
        tuple: (C, clss)
            - C is a numpy.ndarray of shape (k, d) containing the centroids
            - clss is a numpy.ndarray of shape (n,) with cluster assignments
    """
    if not hasattr(X, "shape") or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
