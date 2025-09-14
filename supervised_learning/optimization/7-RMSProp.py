#!/usr/bin/env python3
"""This module includes the function
that updates a variable using the
RMSProp optimization algorithm"""

import numpy as np

def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Args:
        alpha is the learning rate
        beta2 is the RMSProp weight
        epsilon is a small number to avoid division by zero
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        s is a numpy.ndarray containing the previous second moment
    Returns:
        the updated variable and the new moment, respectively
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
