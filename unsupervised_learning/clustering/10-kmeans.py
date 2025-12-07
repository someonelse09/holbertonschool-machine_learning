#!/usr/bin/env/python3
"""This module includes the function kmeans
that performs K-means on a dataset"""

import sklearn.cluster


def kmeans(X, k):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters
        The only import you are allowed to use is import sklearn.cluster
    Returns:
        C, clss
        C is a numpy.ndarray of shape (k, d)
         containing the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the
         index of the cluster in C that each data point belongs to
    """
    # Create KMeans object and fit to data
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)

    # Get cluster centers (centroids)
    C = kmeans_model.cluster_centers_

    # Get cluster assignments for each point
    clss = kmeans_model.labels_

    return C, clss
