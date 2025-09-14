#!/usr/bin/env python3
"""This module includes the function named create_mini_batch"""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """This function creates mini-batches to be used
     for training a neural network using mini-batch gradient descent"""
    X, Y = shuffle_data(X, Y)
    m = Y.shape[0]
    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
