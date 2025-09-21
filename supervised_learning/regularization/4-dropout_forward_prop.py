#!/usr/bin/env python3
"""This module includes the function
that conducts forward propagation using Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Args:
        X is a numpy.ndarray of shape
         (nx, m) containing the input data for the network
        nx is the number of input features
        m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        L the number of layers in the network
        keep_prob is the probability that a node will be kept
        All layers except the last should use the tanh activation function
        The last layer should use the softmax activation function
    Returns:
        a dictionary containing the outputs of each layer
         and the dropout mask used on each layer (see example for format)
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        w = weights[f"W{i}"]
        b = weights[f"b{i}"]
        previous_A = cache[f"A{i - 1}"]
        Z = np.dot(w, previous_A) + b
        
        # Last layer uses softmax activation
        if i == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            # Apply Dropout (to Hidden Layers, not Output layer)
            # Creating Dropout mask
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            D = D.astype(int)

            # Apply Dropout mask and scale
            A = A * D
            A = A / keep_prob
            # Store dropout mask
            cache[f"D{i}"] = D
        cache[f"A{i}"] = A
    return cache
