#!/usr/bin/env python3
"""This module includes the function that updates
the weights and biases of a neural network
using gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Args:
        Y is a one-hot numpy.ndarray of shape
         (classes, m) that contains the correct labels for the data
        classes is the number of classes
        m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs of each layer of the neural network
        alpha is the learning rate
        lambtha is the L2 regularization parameter
        L is the number of layers of the network
        The neural network uses tanh activations
         on each layer except the last, which uses a softmax activation
        The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    AL = cache[f"A{L}"]
    dZ = AL - Y
    for i in range(L, 0, -1):
        previous_A = cache[f"A{i - 1}"]
        w = weights[f"W{i}"]
        b = weights[f"b{i}"]
        dw = (dZ @ previous_A.T) / m + (lambtha / (2 * m)) * w
        db = np.sum(dZ, axis=1, keepdims=True) / m

        w -= alpha * dw
        b -= alpha * db
        if i > 1:
            previous_A = cache[f"A{i - 1}"]
            dZ = (w.T @ dZ) * (1 - previous_A ** 2)
