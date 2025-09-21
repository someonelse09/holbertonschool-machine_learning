#!/usr/bin/env python3
"""This module includes the function that
calculates the cost of a neural network with L2 regularization"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Args:
        cost is the cost of the network without L2 regularization
        lambtha is the regularization parameter
        weights is a dictionary of the weights and
         biases (numpy.ndarrays) of the neural network
        L is the number of layers in the neural network
        m is the number of data points used
    Returns:
        the cost of the network accounting for L2 regularization
    """
    lasso_sum = 0
    for i in range(1, L + 1):
        w = weights[f"W{i}"]
        lasso_sum += np.sum(np.square(w))
    lasso_cost = cost + (lambtha / (2 * m)) * lasso_sum
    return lasso_cost
