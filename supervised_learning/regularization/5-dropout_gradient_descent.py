#!/usr/bin/env python3
"""This module includes the function that updates the weights
of a neural network with Dropout regularization using gradient descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Args:
        Y is a one-hot numpy.ndarray of shape
         (classes, m) that contains the correct labels for the data
        classes is the number of classes
        m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs and
         dropout masks of each layer of the neural network
        alpha is the learning rate
        keep_prob is the probability that a node will be kept
        L is the number of layers of the network
        All layers use the tanh activation function
         except the last, which uses the softmax activation function
        The weights of the network should be updated in place
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        previous_A = cache[f"A{i - 1}"]
        w = weights[f"W{i}"]
        dw = (1 / m) * (dZ @ previous_A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            dprevious_A = (w.T @ dZ)
            previous_D = cache[f"D{i - 1}"]
            dprevious_A = dprevious_A * previous_D / keep_prob
            dZ = dprevious_A * (1 - previous_A ** 2)
        weights[f"W{i}"] -= alpha * dw
        weights[f"b{i}"] -= alpha * db
